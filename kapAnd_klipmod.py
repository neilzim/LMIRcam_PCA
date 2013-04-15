#! /usr/bin/env python
"""

Implement KLIP point source forward model on LMIRcam kappa And ADI data.

"""
import pdb
import numpy as np
from multiprocessing import Process, Queue, cpu_count
import time as time
import pyfits
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.optimize import *
import sys
import os
import shelve
import matplotlib.pyplot as plt
import matplotlib

def superpose_srcmodel(data_img, srcmodel_img, srcmodel_destxy, srcmodel_centxy = None, rolloff_rad = None):
    assert(len(data_img.shape) == 2)
    assert(len(srcmodel_img.shape) == 2)
    assert( srcmodel_destxy[0] < data_img.shape[1] and srcmodel_destxy[1] < data_img.shape[0]\
            and min(srcmodel_destxy) >= 0 )
    if srcmodel_centxy == None:
        srcmodel_centxy = ((srcmodel_img.shape[0] - 1.)/2., (srcmodel_img.shape[1] - 1.)/2.)
    subpix_xyoffset = np.array( [(srcmodel_destxy[0] - srcmodel_centxy[0])%1.,\
                                 (srcmodel_destxy[1] - srcmodel_centxy[1])%1.] )
    if abs(round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])) == 1: # jump in pixel containing x center
        subpix_xyoffset[0] -= round(srcmodel_centxy[0] + subpix_xyoffset[0]) - round(srcmodel_centxy[0])
    if abs(round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])) == 1: # jump in pixel containing y center
        subpix_xyoffset[1] -= round(srcmodel_centxy[1] + subpix_xyoffset[1]) - round(srcmodel_centxy[1])
    #print "subpix_offset: ", subpix_xyoffset

    if rolloff_rad:
        Y, X = np.indices(srcmodel_img.shape)
        Rsqrd = (X - srcmodel_centxy[0])**2 + (Y - srcmodel_centxy[1])**2
        rolloff_arr = np.exp( -(Rsqrd / rolloff_rad**2)**2 )
        srcmodel_img *= rolloff_arr
    shifted_srcmodel_img = shift(input = srcmodel_img, shift = subpix_xyoffset[::-1], order=3)

    srcmodel_BLcorneryx = np.array( [round(srcmodel_destxy[1]) - round(srcmodel_centxy[1]),
                                     round(srcmodel_destxy[0]) - round(srcmodel_centxy[0])], dtype=np.int)
    srcmodel_TRcorneryx = srcmodel_BLcorneryx + np.array(srcmodel_img.shape)
    super_BLcorneryx = np.amax(np.vstack((srcmodel_BLcorneryx, np.zeros(2))), axis=0)
    super_TRcorneryx = np.amin(np.vstack((srcmodel_TRcorneryx, np.array(data_img.shape))), axis=0)
    BLcropyx = super_BLcorneryx - srcmodel_BLcorneryx
    TRcropyx = srcmodel_TRcorneryx - super_TRcorneryx
    super_img = data_img.copy()
    super_img[super_BLcorneryx[0]:super_TRcorneryx[0],\
              super_BLcorneryx[1]:super_TRcorneryx[1]] +=\
            shifted_srcmodel_img[BLcropyx[0]:srcmodel_img.shape[0]-TRcropyx[0],\
                                 BLcropyx[1]:srcmodel_img.shape[1]-TRcropyx[1]]
    return super_img 

def reconst_zone(data_vec, pix_table, img_dim):
    reconstrd_img = np.zeros(img_dim)
    for i, pix_val in enumerate(data_vec.flat):
        row = pix_table[0][i] 
        col = pix_table[1][i]
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

def mp_eval_adiklip_srcmodel(p, N_proc, op_fr, adiklip_config, adiklip_data, srcmodel):
    if op_fr == None:
        op_fr = adiklip_config['op_fr']
    N_op_fr = len(op_fr)
    cost_queue = Queue()
    total_sumofsq_cost = 0

    mp_op_fr = list()
    for i in range(N_proc):
        chunk_size = round(float(N_op_fr) / N_proc)
        fr_ind_beg = i * chunk_size
        if i != N_proc - 1:
            fr_ind_end = fr_ind_beg + chunk_size
        else:
            fr_ind_end = N_op_fr
        mp_op_fr.append(op_fr[fr_ind_beg:fr_ind_end])

    for i in range(N_proc):
        task = Process(target = eval_adiklip_srcmodel, args = (p, mp_op_fr[i], adiklip_config,
                                                               adiklip_data, srcmodel, cost_queue))
        task.start()

    for i in range(N_proc):
        total_sumofsq_cost += cost_queue.get()

    return total_sumofsq_cost

def lnprob_adiklip_srcmodel(p, N_proc, op_fr, adiklip_config, adiklip_data, srcmodel):
    img_shape = adiklip_config['fr_shape']
    if abs(p[1]) < (img_shape[1] - 1.)/2. and abs(p[2]) < (img_shape[0] - 1)/2.:
        cost = mp_eval_adiklip_srcmodel(p = p, N_proc = N_proc, op_fr = op_fr, adiklip_config = adiklip_config,
                                        adiklip_data = adiklip_data, srcmodel = srcmodel)
        lnprob = -cost/2.
    else:
        lnprob = np.finfo(np.float).min
    return lnprob

def eval_adiklip_srcmodel(p, op_fr, adiklip_config, adiklip_data, srcmodel, cost_queue=None, res_cube_fname=None):
    fr_shape = adiklip_config['fr_shape']
    parang_seq = adiklip_config['parang_seq']
    mode_cut = adiklip_config['mode_cut']
    track_mode = adiklip_config['track_mode']
    op_rad = adiklip_config['op_rad']
    op_az = adiklip_config['op_az']
    ref_table = adiklip_config['ref_table']
    zonemask_table_1d = adiklip_config['zonemask_table_1d']
    zonemask_table_2d = adiklip_config['zonemask_table_2d']
    N_op_fr = len(op_fr)
    N_fr = len(parang_seq)

    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    srcmodel_cent_xy = ((srcmodel.shape[1] - 1)/2., (srcmodel.shape[0] - 1)/2.)
    amp = p[0]
    deltax = p[1]
    deltay = p[2]
    theta = np.arctan2(deltay, deltax)
    rho = np.sqrt(deltax**2 + deltay**2)
    abframe_theta_seq = [theta - np.deg2rad(parang) for parang in parang_seq[op_fr]]
    abframe_xy_seq = [( cent_xy[0] + rho*np.cos(t),
                        cent_xy[1] + rho*np.sin(t) ) for t in abframe_theta_seq]
    abframe_synthsrc_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    sumofsq_cost = 0.
    if res_cube_fname:
        if os.path.exists(res_cube_fname) == False:
            res_cube = np.zeros((N_fr, fr_shape[0], fr_shape[1]))
            res_cube_hdu = pyfits.PrimaryHDU(res_cube.astype(np.float32))
            res_cube_hdu.writeto(res_cube_fname)
        res_cube_hdulist = pyfits.open(res_cube_fname, mode='update')
        res_cube = res_cube_hdulist[0].data
    for op_fr_ind, fr_ind in enumerate(op_fr):
        if abframe_xy_seq[op_fr_ind][0] >= fr_shape[1] or abframe_xy_seq[op_fr_ind][1] >= fr_shape[0] or min(abframe_xy_seq[op_fr_ind]) < 0:
            print 'bad dest:', abframe_xy_seq[op_fr_ind]
            print 'p:', p[0:3]
            return np.finfo(np.float).max 
        abframe_synthsrc_cube[op_fr_ind,:,:] = superpose_srcmodel(data_img = np.zeros(fr_shape), srcmodel_img = amp*srcmodel,
                                                                  srcmodel_destxy = abframe_xy_seq[op_fr_ind], srcmodel_centxy = srcmodel_cent_xy)
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                Pmod = np.ravel(abframe_synthsrc_cube[op_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                Pmod -= np.mean(Pmod)
                Z = klip_data[fr_ind][rad_ind][az_ind]['Z'][:,:]
                Projmat = np.dot(Z.T, Z)
                F = klip_data[fr_ind][rad_ind][az_ind]['F']
                Pmod_proj = np.dot(Pmod, Projmat)
                res_vec = F - Pmod + Pmod_proj
                sumofsq_cost += np.sum(res_vec**2)
                if res_cube_fname:
                    res_cube[fr_ind,:,:] += reconst_zone(res_vec, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                    #print "rms(Pmod) = %.1f, rms(Pmod_proj) = %.1f" %\
                    #      (np.sqrt(np.sum(Pmod**2)), np.sqrt(np.sum(Pmod_proj**2)))
    if cost_queue:
        cost_queue.put(sumofsq_cost)
    if res_cube_fname:
        res_cube_hdulist.close()
    return sumofsq_cost

if __name__ == "__main__":
    data_dir = os.path.expanduser('~/Data/LMIRcam/kappaAnd')
    klipsub_result_dir = os.path.expanduser('~/Data/LMIRcam/kappaAnd/klipsub_results')
    klipmod_result_dir = os.path.expanduser('~/Data/LMIRcam/kappaAnd/klipmod_results')
    klipsub_archv_fname = "%s/kapAnd_cut10_delPhi30_klipsub_archive.shelve" % klipsub_result_dir
    #klipsub_archv_fname = "%s/kapAnd_cut10_delPhi20_klipsub_archive.shelve" % klipsub_result_dir
    #klipsub_archv_fname = "%s/kapAnd_cut05_delPhi30_klipsub_archive.shelve" % klipsub_result_dir
    synthpsf_fname = '%s/psf_model.fits' % data_dir
    guess_res_cube_fname = '%s/guess_res_cube.fits' % klipmod_result_dir
    final_res_cube_fname = '%s/final_res_cube.fits' % klipmod_result_dir
    fbf_res_cube_fname = '%s/fbf_res_cube.fits' % klipmod_result_dir

    do_MLE = True
    do_frbyfr_MLE = True
    do_MCMC = False

    assert os.path.exists(klipsub_result_dir)
    assert os.path.exists(klipmod_result_dir)
    assert os.path.exists(klipsub_archv_fname)

    klipsub_archv = shelve.open(klipsub_archv_fname, 'r')
    print 'Opened KLIP subtraction archive %s' % klipsub_archv_fname
    klip_config = klipsub_archv['klip_config']
    klip_data = klipsub_archv['klip_data']
    klipsub_archv.close()

    fr_shape = klip_config['fr_shape']
    parang_seq = klip_config['parang_seq']
    mode_cut = klip_config['mode_cut']
    op_fr = klip_config['op_fr']
    #op_fr = np.arange(43,87)
    op_rad = klip_config['op_rad']
    op_az = klip_config['op_az']
    ref_table = klip_config['ref_table']
    zonemask_table_1d = klip_config['zonemask_table_1d']
    zonemask_table_2d = klip_config['zonemask_table_2d']
    N_op_fr = len(op_fr)
    #
    # Load PSF model, crop, and apply "roll-off" window to keep the edges smooth.
    #
    synthpsf_hdulist = pyfits.open(synthpsf_fname)
    synthpsf_img = synthpsf_hdulist[0].data
    synthpsf_hdulist.close()
    synthpsf_cent_xy = ((synthpsf_img.shape[0] - 1.)/2., (synthpsf_img.shape[1] - 1.)/2.)
    rolloff_rad = 40.
    cropmarg = 1
    crop_synthpsf_img = synthpsf_img[cropmarg:-cropmarg, cropmarg:-cropmarg].copy()
    crop_synthpsf_cent_xy = ((crop_synthpsf_img.shape[0] - 1.)/2., (crop_synthpsf_img.shape[1] - 1.)/2.)
    Y, X = np.indices(crop_synthpsf_img.shape)
    Rsqrd = (X - crop_synthpsf_cent_xy[0])**2 + (Y - crop_synthpsf_cent_xy[1])**2
    crop_synthpsf_img *= np.exp( -(Rsqrd / rolloff_rad**2)**2 )
    print synthpsf_img.shape, crop_synthpsf_img.shape
    print synthpsf_img.sum(), crop_synthpsf_img.sum()
    #
    # Set the source parameters to a reasonable guess.
    #
    cent_xy = ((fr_shape[1] - 1)/2., (fr_shape[0] - 1)/2.)
    ampguess = 600.
    posguess_rho = 97.4
    posguess_theta = 55.6 # CCW of N (up in derotated image)
    #posguess_rho = 100.
    #posguess_theta = 60. # CCW of N (up in derotated image)
    posguess_xy = ( cent_xy[0] + posguess_rho*np.cos(np.deg2rad(posguess_theta + 90)),\
                       cent_xy[1] + posguess_rho*np.sin(np.deg2rad(posguess_theta + 90)) )
    posguess_deltaxy = (posguess_xy[0] - cent_xy[0], posguess_xy[1] - cent_xy[1])
    p0 = np.array([ampguess, posguess_deltaxy[0], posguess_deltaxy[1]])
    p_min = [ampguess*0.1, posguess_deltaxy[0] - 5., posguess_deltaxy[1] - 5.]
    p_max = [ampguess*10., posguess_deltaxy[0] + 5., posguess_deltaxy[1] + 5.]
    p_bounds = [(p_min[i], p_max[i]) for i in range(len(p_min))]
    print "p0:", p0

    N_proc = cpu_count() 
    start_time = time.time()
    #guess_cost = eval_adiklip_srcmodel(p = p0, op_fr = op_fr, adiklip_config = klip_config, adiklip_data = klip_data,\
    #                                   srcmodel = synthpsf_img, res_cube_fname=guess_res_cube_fname)
    guess_cost = mp_eval_adiklip_srcmodel(p = p0, N_proc = N_proc, op_fr = None, adiklip_config = klip_config,
                                          adiklip_data = klip_data, srcmodel = crop_synthpsf_img)
    end_time = time.time()
    exec_time = end_time - start_time
    print "Guess param cost func evaluation = %.1f. Took %dm%02ds to evaluate KLIP source model cost for %d ADI frames" %\
          (guess_cost, int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr)

    if do_MLE:
        #
        # Optimize the flux and position of the source model to fit the KLIP subtraction residuals.
        #
        start_time = time.time()
        #p_sol, final_cost, info = fmin_l_bfgs_b(func = eval_adiklip_srcmodel, x0 = p0,
        #                                        args = (op_fr, klip_config, klip_data, synthpsf_img),
        #                                        approx_grad = True, bounds = p_bounds, factr=1e8, maxfun=100, disp=2)
        p_sol, final_cost, info = fmin_l_bfgs_b(func = mp_eval_adiklip_srcmodel, x0 = p0,
                                                args = (N_proc, op_fr, klip_config, klip_data, crop_synthpsf_img),
                                                approx_grad = True, bounds = p_bounds, factr=1e8, maxfun=100, disp=2)
        end_time = time.time()
        exec_time = end_time - start_time

        print "p_sol:", p_sol
        print "Took %dm%02ds to optimize KLIP source model for %d ADI frames" %\
              (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr)

        eval_adiklip_srcmodel(p = p_sol, op_fr = op_fr, adiklip_config = klip_config, adiklip_data = klip_data,\
                              srcmodel = crop_synthpsf_img, res_cube_fname = final_res_cube_fname)
        if do_frbyfr_MLE:
            #
            # Find the MLE flux and position solution for each individual ADI frame
            #
            print "Doing frame-by-frame MLE of source model..."
            fbf_flux_sol = list()
            p_min_fbf = [p_sol[0]*0.5, p_sol[1] - 5., p_sol[2] - 5.]
            p_max_fbf = [p_sol[0]*2.,  p_sol[1] + 5., p_sol[2] + 5.]
            p_bounds_fbf = [(p_min_fbf[i], p_max_fbf[i]) for i in range(len(p_min_fbf))]
            for f in op_fr:
                p_sol_fbf, final_cost_fbf, info = fmin_l_bfgs_b(func = mp_eval_adiklip_srcmodel, x0 = p_sol,
                                                                args = (N_proc, np.array([f]), klip_config, klip_data, crop_synthpsf_img),
                                                                approx_grad = True, bounds = p_bounds, factr=1e8, maxfun=100, disp=0)
                fbf_flux_sol.append(p_sol_fbf[0])
                #print "\tFrame %d: peak flux %0.2f" % (f, p_sol_fbf[0])
                eval_adiklip_srcmodel(p = p_sol_fbf, op_fr = np.array([f]), adiklip_config = klip_config, adiklip_data = klip_data,\
                                      srcmodel = crop_synthpsf_img, res_cube_fname = fbf_res_cube_fname)
            print "Wrote model residual cube to %s" % fbf_res_cube_fname
            print "mean, median, and std dev of frame-by-frame flux solution:", np.mean(fbf_flux_sol), np.median(fbf_flux_sol), np.std(fbf_flux_sol)

    if do_MCMC:
        #
        # Use MCMC to sample the posterior probability distribution of flux and position
        #
        import emcee
        print "Starting the MCMC calculation..."
        start_time = time.time()
        N_dim = 3
        #N_iter = 400
        N_iter = 100
        N_burn = 100
        N_walkers = 30
        if do_MLE:
            p_init = np.transpose( np.vstack(( p_sol[0] + 300 * (0.5 - np.random.rand(N_walkers)),
                                               p_sol[1] + 2 * (0.5 - np.random.rand(N_walkers)),
                                               p_sol[2] + 2 * (0.5 - np.random.rand(N_walkers)) )) )
        else:
            p_init = np.transpose( np.vstack(( p0[0] + 300 * (0.5 - np.random.rand(N_walkers)),
                                               p0[1] + 2 * (0.5 - np.random.rand(N_walkers)),
                                               p0[2] + 2 * (0.5 - np.random.rand(N_walkers)) )) )
        sampler = emcee.EnsembleSampler(N_walkers, N_dim, lnprob_adiklip_srcmodel,
                                        args=[N_proc, op_fr, klip_config, klip_data, crop_synthpsf_img])
        pos, prob, state = sampler.run_mcmc(p_init, N_burn)
        sampler.reset()
        sampler.run_mcmc(pos, N_iter)
        end_time = time.time()
        exec_time = end_time - start_time
        print "Took %02dh%02dm%02ds to run MCMC over the flux and position parameter space" %\
              (int(exec_time/60./60.), int((exec_time - 60*60*int(exec_time/60./60.))/60.), exec_time - 60*int(exec_time/60.))
        T_struct = time.localtime()
        mcmc_archv_fname = "%s/kapAnd_MCMC_archive_Nwalk%03d_Niter%03d_%04d-%02d-%02d_%02dh-%02dm.shelve" %\
                           (klipmod_result_dir, N_walkers, N_iter,
                            T_struct.tm_year, T_struct.tm_mon, T_struct.tm_mday,
                            T_struct.tm_hour, T_struct.tm_min)
        mcmc_archv = shelve.open(mcmc_archv_fname)
        mcmc_archv['p_init'] = p_init
        mcmc_archv['N_dim'] = N_dim
        mcmc_archv['N_iter'] = N_iter
        mcmc_archv['N_burn'] = N_burn
        mcmc_archv['N_walkers'] = N_walkers
        mcmc_archv['sampler'] = sampler
        mcmc_archv.close()
        print "Stored the MCMC results in %s" % mcmc_archv_fname

        plt.hist(sampler.flatchain[:,0])
        plt.show()
