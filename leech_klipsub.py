#! /usr/bin/env python
"""

Carry out PSF subtraction on LMIRcam kappa And ADI data set using principal
component analysis/K-L.

"""

import numpy as np
import time as time
import pyfits
from scipy.ndimage.interpolation import *
from scipy.interpolate import *
from scipy.ndimage.filters import *
from scipy.io.idl import readsav
import sys
import os
import pdb
import shelve
import matplotlib.pyplot as plt
import matplotlib.colors

def get_radius_sqrd(s, c=None):
    if c is None:
        c = (0.5*float(s[0] - 1),  0.5*float(s[1] - 1))
    y, x = np.indices(s)
    rsqrd = (x - c[0])**2 + (y - c[1])**2
    return rsqrd

def get_angle(s, c=None):
    if c is None:
        c = (0.5*float(s[0] - 1),  0.5*float(s[1] - 1))
    y, x = np.indices(s)
    theta = np.arctan2(y - c[1], x - c[0])
    # Change the angle range from [-pi, pi] to [0, 360]
    theta_360 = np.where(np.greater_equal(theta, 0), np.rad2deg(theta), np.rad2deg(theta + 2*np.pi))
    return theta_360

def reconst_zone_cube(data_mat, pix_table, cube_dim):
    reconstrd_cube = np.zeros(cube_dim)
    assert(data_mat.shape[0] >= cube_dim[0])
    for fr_ind in range(cube_dim[0]):
        for i, pix_val in enumerate(data_mat[fr_ind, :].flat):
            row = pix_table[0][i]
            col = pix_table[1][i]
            #if np.isreal(pix_val) == False:
            #    print "reconst_zone_cube: warning - complex valued pixel"
            reconstrd_cube[fr_ind, row, col] = pix_val
    return reconstrd_cube

def reconst_zone(data_vec, pix_table, img_dim):
    reconstrd_img = np.zeros(img_dim)
    for i, pix_val in enumerate(data_vec.flat):
        row = pix_table[0][i]
        col = pix_table[1][i]
        #if np.isreal(pix_val) == False:
        #    print "reconst_zone_cube: warning - complex valued pixel"
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

def load_leech_adiseq(fname_root, N_fr, old_xycent, outer_search_rad):
    cropped_cube = np.zeros((N_fr, 2*outer_search_rad, 2*outer_search_rad))
    subpix_xyoffset = np.array( [0.5 - old_xycent[0]%1., 0.5 - old_xycent[1]%1.] )
    print 'load_leech_adiseq: subpix_xyoffset = %0.2f, %0.2f' % (subpix_xyoffset[0], subpix_xyoffset[1])
    shifted_xycent = ( old_xycent[0] + subpix_xyoffset[0], old_xycent[1] + subpix_xyoffset[1] )

    for i in range(N_fr):
        img_fname = fname_root + '%d.fits' % i
        img_hdulist = pyfits.open(img_fname, 'readonly')
        img = img_hdulist[0].data
        old_width = img.shape[1]
        shifted_img = shift(input = img, shift = subpix_xyoffset[::-1], order=3)    
        cropped_cube[i, :, :] = shifted_img[ round(shifted_xycent[1]) - outer_search_rad:round(shifted_xycent[1]) + outer_search_rad,
                                             round(shifted_xycent[0]) - outer_search_rad:round(shifted_xycent[0]) + outer_search_rad ].copy()
        img_hdulist.close()
    return cropped_cube

def load_data_cube(datacube_fname, outer_search_rad):
    cube_hdu = pyfits.open(datacube_fname, 'readonly')
    old_width = cube_hdu[0].data.shape[1]
    crop_margin = (old_width - 2*outer_search_rad)/2
    print "Cropping %d-pixel wide frames down to %d pixels."%(old_width, old_width-2*crop_margin)
    cropped_cube = cube_hdu[0].data[:, crop_margin-1:-crop_margin-1, crop_margin-1:-crop_margin-1].copy()
    cube_hdu.close()
    return cropped_cube

def load_and_clean_data_cube(datacube_fname, outer_search_rad, replace_bad):
    cube_hdu = pyfits.open(datacube_fname, 'readonly')
    N_fr = cube_hdu[0].data.shape[0]
    old_height = cube_hdu[0].data.shape[1]
    old_width = cube_hdu[0].data.shape[2]
    crop_margin = (old_width - 2*outer_search_rad)/2
    sci_xbeg = crop_margin - 1
    sci_xend = -crop_margin - 1
    sci_ybeg = sci_xbeg
    sci_yend = sci_xend
    print "Cropping %d-pixel wide frames down to %d pixels."%(old_width, old_width-2*crop_margin)
    cropped_cube = cube_hdu[0].data[:, sci_ybeg:sci_yend, sci_xbeg:sci_xend].copy()
    #
    # ad hoc/hard-coded instructions for cleaning kappa And detector artifacts
    #
    clean_cropped_cube = np.zeros(cropped_cube.shape)
    use_bottom_edge = [True] * N_fr
    colsamp_height = [50] * N_fr
    colsamp_height[24:32] = [80 for i in range(24,32)]
    colsamp_height[48:72] = [80 for i in range(48,72)]
    use_bottom_edge[40:48] = [False for i in range(40,48)]
    use_bottom_edge[56:64] = [False for i in range(56,64)]
    use_bottom_edge[80:N_fr] = [False for i in range(80,N_fr)]
    gradsamp_xbeg = 2
    gradsamp_xend = 120
    gradsamp_ybeg = 320
    #for fr_ind in range(80,81):
    #for fr_ind in range(24,25):
    for fr_ind in range(N_fr):
        img = cube_hdu[0].data[fr_ind,:,:].copy()
        if use_bottom_edge[fr_ind]:
            test_row = 0
            while np.isnan(img[test_row, old_width/2]):
                test_row += 1
            colsamp_ybeg = test_row
            colsamp_yend = colsamp_ybeg + colsamp_height[fr_ind]
            img -= np.median(img[colsamp_ybeg:colsamp_yend,:], axis=0)
            #img_hdu = pyfits.PrimaryHDU(img)
            #img_hdu.writeto("%s/test_clean_img.fits" % result_dir, clobber=True)
        else:
            test_row = -1
            while np.isnan(img[test_row, old_width/2]):
                test_row -= 1
            colsamp_yend = test_row - 1 # 2-pixel margin at top edge of detector
            gradsamp_yend = colsamp_yend
            colsamp_ybeg = test_row - colsamp_height[fr_ind]
            #print "fr %d: top good row is %d" % (fr_ind, colsamp_yend)
            gradsamp_box = img[gradsamp_ybeg:gradsamp_yend, gradsamp_xbeg:gradsamp_xend].copy()
            gradsamp_box -= np.median(gradsamp_box, axis=0)
            gradsamp_vec = np.mean(gradsamp_box, axis=1)
            rowpts = np.arange(gradsamp_vec.shape[0])
            #spline = UnivariateSpline(rowpts, gradsamp_vec, s=10**6)
            #spline_fit = spline(rowpts)
            cubic_op = np.poly1d(np.polyfit(rowpts, gradsamp_vec, 3))
            cubic_fit = cubic_op(rowpts)
            grad_correct = (cubic_fit - cubic_fit[0]).reshape((-1,1))
            #plt.plot(gradsamp_vec)
            #plt.plot(cubic_fit)
            #plt.show()
            img[gradsamp_ybeg:gradsamp_yend,:] -= grad_correct
            #img_hdu = pyfits.PrimaryHDU(img)
            #img_hdu.writeto("%s/test_clean_img_step1.fits" % result_dir, clobber=True)
            img -= np.median(img[colsamp_ybeg:colsamp_yend,:], axis=0)
            #clean_img_hdu = pyfits.PrimaryHDU(clean_img)
            #clean_img_hdu.writeto("%s/test_clean_img_step2.fits" % result_dir, clobber=True)
        #img = median_filter(img, size = (2,2))
        clean_cropped_cube[fr_ind,:,:] = img[sci_ybeg:sci_yend, sci_xbeg:sci_xend]
    #
    # Replace bad pixels. The hard-coded list, written below, was identified by hand. Coordinates are zero-based x,y pairs.
    #
    if replace_bad == True:
        bad_list = [(177, 56), (179, 44), (179, 45), (209, 56), (201, 76), (202, 76), (201, 77), (202, 76), (202, 77),
                    (214, 82), (215, 81), (215, 82), (201, 95), (191, 100), (192, 99), (192, 100), (210, 112), (210, 113),
                    (196, 134), (196, 135), (184, 122), (201, 146), (201, 147), (182, 151), (182, 152),
                    (181, 97), (182, 97), (182, 98), (216, 126), (217, 126), (216, 127), (217, 127),
                    (206, 116), (212, 104), (193, 63), (205, 65), (193, 163), (175, 170), (203, 161), (204, 161), (203, 160),
                    (200, 109), (200, 108), (209, 124), (187, 167), (188, 55), (189, 55), (188, 56), (189, 56),
                    (216, 126), (217, 126), (216, 127), (217, 127), (210, 79), (211, 79), (192, 46), (193, 46), (200, 77), (200, 78),
                    (214, 82), (181, 152), (180, 147), (205, 92), (205, 93), (210, 70)]
        #thresh = 1000.
        thresh = 800.
        neighb_reach = 1
        for fr_ind in range(N_fr):
            for bad_pix in bad_list:
                if clean_cropped_cube[fr_ind, bad_pix[1], bad_pix[0]] > thresh:
                    goodneighb_list = list()
                    for x in range(bad_pix[0] - neighb_reach, bad_pix[0] + neighb_reach + 1):
                        for y in range(bad_pix[1] - neighb_reach, bad_pix[1] + neighb_reach + 1):
                            if x > 0 and x < clean_cropped_cube.shape[2] and \
                               y > 0 and y < clean_cropped_cube.shape[1] and (x, y) not in bad_list:
                                goodneighb_list.append( clean_cropped_cube[fr_ind, y, x] )
                    clean_cropped_cube[fr_ind, bad_pix[1], bad_pix[0]] = np.median( goodneighb_list )

    cropped_cube_hdu = pyfits.PrimaryHDU(cropped_cube.astype(np.float32))
    cropped_cube_hdu.writeto("%s/kapAnd_cropped_cube.fits" % result_dir, clobber=True)
    clean_cropped_cube_hdu = pyfits.PrimaryHDU(clean_cropped_cube.astype(np.float32))
    clean_cropped_cube_hdu.writeto("%s/kapAnd_cleaned_cropped_cube.fits" % result_dir, clobber=True)
    cube_hdu.close()
    return clean_cropped_cube

def get_ref_and_pix_tables(xycent, fr_shape, N_fr, op_fr, N_rad, R_inner, R_out, op_rad,
                           N_az, op_az, parang_seq, fwhm, min_refgap_fac, track_mode, diagnos_stride):
    #
    # Determine table of references for each frame, and form search zone pixel masks (1-D and 2-D formats).
    #
    print "Search zone scheme:"
    if track_mode:
        print "\tTrack mode ON"
    else:
        print "\tTrack mode OFF"
    print "\tR_inner:", R_inner, "; R_out:", R_out
    print "\tPhi_0, DPhi, N_az:", Phi_0, DPhi, N_az
    print "\tmode_cut:", mode_cut
    for rad_ind in op_rad:
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        if track_mode:
            min_refang = DPhi[rad_ind]/2.
        else:
            min_refang = np.arctan(min_refgap_fac[rad_ind]*fwhm/((R1 + R2)/2))*180/np.pi
        print "\trad_ind = %d: min_refang = %0.2f deg" % (rad_ind, min_refang)
    print ""

    if xycent == None:
        xycent = ((fr_width - 1)/2., (fr_width - 1)/2.)
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()
    angle_vec = get_angle(fr_shape, xycent).ravel()
    zonemask_table_1d = [[[None]*N_az[r] for r in range(N_rad)] for i in range(N_fr)]
    zonemask_table_2d = [[[None]*N_az[r] for r in range(N_rad)] for i in range(N_fr)]
    ref_table = [[list() for r in range(N_rad)] for i in range(N_fr)]

    for fr_ind in op_fr:
        for rad_ind in op_rad:
            R2 = R_out[rad_ind]
            zonemask_radlist_1d = list()
            zonemask_radlist_2d = list()
            if rad_ind == 0:
                R1 = R_inner
            else:
                R1 = R_out[rad_ind-1]
            if track_mode:
                Phi_beg = (Phi_0[rad_ind] - DPhi[rad_ind]/2. + parang_seq[0] - parang_seq[fr_ind]) % 360.
            else:
                Phi_beg = (Phi_0[rad_ind] - DPhi[rad_ind]/2.) % 360.
            Phi_end = [ (Phi_beg + i * DPhi[rad_ind]) % 360. for i in range(1, N_az[rad_ind]) ]
            Phi_end.append(Phi_beg)
            if track_mode:
                min_refang = DPhi[rad_ind]/2.
            else:
                min_refang = np.arctan(min_refgap_fac[rad_ind]*fwhm/((R1 + R2)/2))*180/np.pi
            ref_table[fr_ind][rad_ind] = np.where(np.greater_equal(np.abs(parang_seq - parang_seq[fr_ind]), min_refang))[0]
            if fr_ind%diagnos_stride == 0:
                print "\tFrame %d/%d, annulus %d/%d: %d valid reference frames." %\
                      (fr_ind+1, N_fr, rad_ind+1, N_rad, len(ref_table[fr_ind][rad_ind]))
            if len(ref_table[fr_ind][rad_ind]) < 1:
                print "Zero valid reference frames for fr_ind = %d, rad_ind = %d." % (fr_ind, rad_ind)
                print "The par ang of this frame is %0.2f deg; min_refang = %0.2f deg. Forced to exit." % (parang_seq[fr_ind], min_refang)
                sys.exit(-1)
                
            for az_ind in op_az[rad_ind]:
                Phi2 = Phi_end[az_ind]
                if az_ind == 0:
                    Phi1 = Phi_beg
                else:
                    Phi1 = Phi_end[az_ind-1]
                if Phi1 < Phi2:
                    mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                            np.greater(rad_vec, R1),\
                                            np.less_equal(angle_vec, Phi2),\
                                            np.greater(angle_vec, Phi1)))
                else: # azimuthal region spans phi = 0
                    rad_mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                                np.greater(rad_vec, R1)))
                    az_mask_logic = np.vstack((np.less_equal(angle_vec, Phi2),\
                                               np.greater(angle_vec, Phi1)))
                    mask_logic = np.vstack((np.any(az_mask_logic, axis=0),\
                                            np.all(rad_mask_logic, axis=0)))
                zonemask_1d = np.nonzero( np.all(mask_logic, axis = 0) )[0]
                zonemask_2d = np.nonzero( np.all(mask_logic, axis = 0).reshape(fr_shape) )
                zonemask_table_1d[fr_ind][rad_ind][az_ind] = zonemask_1d
                zonemask_table_2d[fr_ind][rad_ind][az_ind] = zonemask_2d
                if zonemask_1d.shape[0] < len(ref_table[fr_ind][rad_ind]):
                    print "get_ref_table: warning - size of search zone for frame %d, rad_ind %d, az_ind %d is %d < %d, the # of ref frames for this annulus" %\
                          (fr_ind, rad_ind, az_ind, zonemask_1d.shape[0], len(ref_table[fr_ind][rad_ind]))
                    print "This has previously resulted in unexpected behavior, namely a reference covariance matrix that is not positive definite."
    for rad_ind in op_rad:
        num_ref = [len(ref_table[f][rad_ind]) for f in op_fr]
        print "annulus %d/%d: min, median, max number of ref frames = %d, %d, %d" %\
              ( rad_ind+1, N_rad, min(num_ref), np.median(num_ref), max(num_ref) )
    print ""
    return ref_table, zonemask_table_1d, zonemask_table_2d

def do_klip_subtraction(data_cube, config_dict, result_dict, result_dir, diagnos_stride=40, store_klbasis=False):
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    N_fr = len(parang_seq)
    mode_cut = config_dict['mode_cut']
    track_mode = config_dict['track_mode']
    op_fr = config_dict['op_fr']
    N_op_fr = len(op_fr)
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    ref_table = config_dict['ref_table']
    zonemask_table_1d = config_dict['zonemask_table_1d']
    zonemask_table_2d = config_dict['zonemask_table_2d']

    klipsub_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    klippsf_cube = klipsub_cube.copy()
    derot_klipsub_cube = klipsub_cube.copy()

    start_time = time.time()
    for i, fr_ind in enumerate(op_fr):
        # Loop over operand frames
        if fr_ind%diagnos_stride == 0:
            klbasis_cube = np.zeros((max(mode_cut), fr_shape[0], fr_shape[1]))
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                I = np.ravel(data_cube[fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy() 
                R = np.zeros((ref_table[fr_ind][rad_ind].shape[0], zonemask_table_1d[fr_ind][rad_ind][az_ind].shape[0]))
                for j, ref_fr_ind in enumerate(ref_table[fr_ind][rad_ind]):
                    R[j,:] = np.ravel(data_cube[ref_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                if klip_mode == True: # following Soummer et al. 2012
                    if mean_sub == True:
                        I_mean = np.mean(I)
                        I -= I_mean
                        R -= R.mean(axis=1).reshape(-1, 1)
                    Z, sv, N_modes = get_klip_basis(R = R, cutoff = mode_cut[rad_ind])
                else: # following Amara et al. 2012
                    if mean_sub == True:
                        #I -= np.append(R, I, axis=0).mean(axis=0)
                        #R -= np.append(R, I, axis=0).mean(axis=0)
                        I_mean = I.mean()
                        I -= I_mean
                        R -= R.mean(axis=1).reshape(-1, 1)
                    Z, sv = get_pca_basis(R = R, cutoff = mode_cut[rad_ind])
                #if fr_ind % diagnos_stride == 0:
                #    print "Frame %d/%d, annulus %d/%d, sector %d/%d:" %\
                #          (fr_ind+1, N_fr, rad_ind+1, N_rad, az_ind+1, N_az[rad_ind])
                #    print "\tForming PSF estimate..."
                Projmat = np.dot(Z.T, Z)
                I_proj = np.dot(I, Projmat)
                F = I - I_proj
                if np.all(np.isreal(F)) == False:
                    print "do_klip_subtraction: image projection for fr_ind = %d contains complex value" % fr_ind
                klipsub_cube[i,:,:] += reconst_zone(F, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                klippsf_cube[i,:,:] += reconst_zone(I_proj + I_mean, zonemask_table_2d[fr_ind][rad_ind][az_ind], fr_shape)
                if store_archv:
                    klip_data[fr_ind][rad_ind][az_ind]['I'] = I
                    klip_data[fr_ind][rad_ind][az_ind]['I_mean'] = I_mean
                    klip_data[fr_ind][rad_ind][az_ind]['Z'] = Z
                    klip_data[fr_ind][rad_ind][az_ind]['sv'] = sv
                    #klip_data[fr_ind][rad_ind][az_ind]['Projmat'] = Projmat
                    klip_data[fr_ind][rad_ind][az_ind]['I_proj'] = I_proj
                    klip_data[fr_ind][rad_ind][az_ind]['F'] = F
                if fr_ind % diagnos_stride == 0:
                    klbasis_cube[:N_modes,:,:] += reconst_zone_cube(Z, zonemask_table_2d[fr_ind][rad_ind][az_ind],
                                                                    cube_dim = (N_modes, fr_shape[0], fr_shape[1]))
                    if mean_sub == False:
                        I_mean = 0
                    print "Frame %d (%d total in data cube), annulus %d/%d, sector %d/%d: RMS before/after sub: %0.2f / %0.2f" %\
                          (fr_ind+1, N_fr, rad_ind+1, len(op_rad), az_ind+1, len(op_az[rad_ind]),\
                           np.sqrt(np.mean((I + I_mean)**2)), np.sqrt(np.mean(F**2)))
        # De-rotate the KLIP-subtracted image
        derot_klipsub_img = rotate(klipsub_cube[i,:,:], -parang_seq[fr_ind], reshape=False)
        derot_klipsub_cube[i,:,:] = derot_klipsub_img
        if fr_ind % diagnos_stride == 0:
            print "***** Frame %d/%d has been PSF-sub'd and derotated. *****" % (fr_ind+1, N_fr)
            if store_klbasis == True:
                klbasis_cube_hdu = pyfits.PrimaryHDU(klbasis_cube.astype(np.float32))
                klbasis_cube_hdu.writeto("%s/klbasis_fr%03d.fits" % (result_dir, fr_ind), clobber=True)
    end_time = time.time()
    exec_time = end_time - start_time
    time_per_frame = exec_time/N_op_fr
    print "Took %dm%02ds to KLIP-subtract %d frames (%0.2f s per frame).\n" %\
          (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr, time_per_frame)
    return klipsub_cube, klippsf_cube, derot_klipsub_cube

def get_pca_basis(R, cutoff):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[0:cutoff, :], sv

def get_klip_basis(R, cutoff):
    #np.linalg.cholesky(np.dot(R, np.transpose(R)))
    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
    sv = np.sqrt(w[sort_ind]).reshape(-1,1) #column of ranked singular values
    Z = np.dot(1./sv*np.transpose(V[:, sort_ind]), R)
    #for i in range(w.shape[0]):
    #    if w[i] < 0:
    #        print "negative eigenval: w[%d] = %g" % (i, w[i])
    #        #pdb.set_trace()
    N_modes = min([cutoff, Z.shape[0]])
    return Z[0:N_modes, :], sv, N_modes

def get_residual_stats(config_dict, Phi_0, coadd_img, med_img, xycent=None):
    if xycent == None:
        xycent = ((fr_width - 1)/2., (fr_width - 1)/2.)
    fr_shape = config_dict['fr_shape']
    parang_seq = config_dict['parang_seq']
    op_rad = config_dict['op_rad']
    op_az = config_dict['op_az']
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, xycent)).ravel()
    
    Phi_0_derot = (Phi_0 + parang_seq[0]) % 360.
    coadd_annular_rms = list()
    zonal_rms = [[None]*N_az[r] for r in range(N_rad)]
    print "RMS counts in KLIP results:"
    for rad_ind in op_rad:
        R2 = R_out[rad_ind]
        if rad_ind == 0:
            R1 = R_inner
        else:
            R1 = R_out[rad_ind-1]
        annular_mask_logic = np.vstack([np.less_equal(rad_vec, R2),\
                                        np.greater(rad_vec, R1)])
        annular_mask = np.nonzero( np.all(annular_mask_logic, axis=0) )[0]
        coadd_annular_rms.append( np.sqrt( np.mean( np.ravel(coadd_img)[annular_mask]**2 ) ) )
        print "\tannulus %d/%d: %.2f in KLIP sub'd, derotated, coadded annlus" % (rad_ind+1, len(op_rad), coadd_annular_rms[-1])
        if len(op_az[rad_ind]) > 1:
            Phi_beg = (Phi_0_derot - DPhi[rad_ind]/2.) % 360.
            Phi_end = [ (Phi_beg + i * DPhi[rad_ind]) % 360. for i in range(1, len(op_az[rad_ind])) ]
            Phi_end.append(Phi_beg)
            for az_ind in op_az[rad_ind]:
                Phi2 = Phi_end[az_ind]
                if az_ind == 0:
                    Phi1 = Phi_beg
                else:
                    Phi1 = Phi_end[az_ind-1]
                if Phi1 < Phi2:
                    mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                            np.greater(rad_vec, R1),\
                                            np.less_equal(angle_vec, Phi2),\
                                            np.greater(angle_vec, Phi1)))
                else: # azimuthal region spans phi = 0
                    rad_mask_logic = np.vstack((np.less_equal(rad_vec, R2),\
                                                np.greater(rad_vec, R1)))
                    az_mask_logic = np.vstack((np.less_equal(angle_vec, Phi2),\
                                               np.greater(angle_vec, Phi1)))
                    mask_logic = np.vstack((np.any(az_mask_logic, axis=0),\
                                            np.all(rad_mask_logic, axis=0)))
                derot_zonemask = np.nonzero( np.all(mask_logic, axis = 0) )[0]
                zonal_rms[rad_ind][az_ind] = np.sqrt( np.mean( np.ravel(coadd_img)[derot_zonemask]**2 ) )
            delimiter = ', '
            print "\tby zone: %s" % delimiter.join(["%.2f" % zonal_rms[rad_ind][a] for a in op_az[rad_ind]])
    print "Peak value in final co-added image: %0.2f" % np.amax(coadd_img)
    print "Peak value in median of de-rotated images: %0.2f" % np.amax(med_img)
    return coadd_annular_rms, zonal_rms

if __name__ == "__main__":
    #
    # Set KLIP parameters
    #
    klip_mode = True 
    mean_sub = True
    #
    # point PCA search zone config
    #
    #track_mode = True
    #mode_cut = [10]
    #R_inner = 85.
    #R_out = [110.]
    #DPhi = [30.]
    ##DPhi = [20.]
    ##DPhi = [72.]
    #Phi_0 = [328.]
    #Phi_0 = [-6.]
    #
    # global PCA search zone config
    #
    track_mode = False
    mode_cut = [30, 30, 30, 20]
    R_inner = 5.
    R_out = [10, 20, 40, 75.]
    DPhi = [360.]*4
    Phi_0 = [0.]*4

    N_rad = len(R_out)
    fwhm = 11.
    #min_refgap_fac = 1.3
    min_refgap_fac = [0.37, 0.7, 1.0, 1.5]
    assert(len(mode_cut) == N_rad == len(DPhi) == len(Phi_0))
    N_az = [ int(np.ceil(360./DPhi[r])) for r in range(N_rad) ]
    #
    # Load data
    #
    data_dir = os.path.expanduser('~/Data/LMIRcam/leech_test/cxsxcbcnlm')
    result_dir = os.path.expanduser('~/Data/LMIRcam/leech_test/cxsxcbcnlm/klipsub_results')
    assert(os.path.exists(data_dir)), 'data_dir %s does not exist' % data_dir
    assert(os.path.exists(result_dir)), 'result_dir %s does not exist' % result_dir
    L_img_fname_root = '%s/l/combined_' % data_dir
    R_img_fname_root = '%s/r/combined_' % data_dir
    parang_L_fname = '%s/l_PA.sav' % data_dir
    parang_R_fname = '%s/r_PA.sav' % data_dir
    parang_seq_L = readsav(parang_L_fname).combined_images_pa
    parang_seq_R = readsav(parang_R_fname).combined_images_pa
    N_fr_L = parang_seq_L.shape[0]
    N_fr_R = parang_seq_R.shape[0]
    xycent_L = (93, 94)
    xycent_R = (91.5, 93.5)
    data_cube_L = load_leech_adiseq(L_img_fname_root, N_fr_L, xycent_L, R_out[-1])
    data_cube_R = load_leech_adiseq(R_img_fname_root, N_fr_R, xycent_R, R_out[-1])
    #Data_cube = load_data_cube(cube_fname, R_out[-1])
    #Data_cube = load_and_clean_data_cube(cube_fname, R_out[-1], replace_bad = False)
    #Data_cube = load_and_clean_data_cube(cube_fname, R_out[-1], replace_bad = True)
    assert(data_cube_L.shape[1:] == data_cube_R.shape[1:])
    fr_shape = data_cube_L.shape[1:]
    fr_width = fr_shape[1]
    N_parang_L = parang_seq_L.shape[0]
    N_parang_R = parang_seq_R.shape[0]
    assert(np.equal(N_fr_L, N_parang_L))
    assert(np.equal(N_fr_R, N_parang_R))
    print "Loaded and cropped the left and right LMIRcam ADI sequences to width %d pixels." % fr_width
    print "Left channel: %d images with parallactic angle range %0.2f to %0.2f deg" % (N_fr_L, parang_seq_L[0], parang_seq_L[-1])
    print "Right channel: %d images with parallactic angle range %0.2f to %0.2f deg" % (N_fr_R, parang_seq_R[0], parang_seq_R[-1])
    
    #R_cube_hdu = pyfits.PrimaryHDU(R_data_cube.astype(np.float32))
    #R_cube_hdu.writeto('R_cube_test.fits')

    #
    # Set additional program parameters
    #
    store_results = True
    #store_results = False
    #store_archv = True
    store_archv = False
    diagnos_stride = 50
    op_fr_L = np.arange(N_fr_L)
    #op_fr_L = np.arange(0, N_fr_L, diagnos_stride)
    #op_fr_L = np.array([12, 15, 20])
    N_op_fr_L = op_fr_L.shape[0]
    op_rad = range(N_rad)
    op_az = [range(N_az[i]) for i in range(N_rad)]
    #op_az = [[0]]
    #op_az = [[0, 6]]
    assert(len(op_rad) == len(op_az) == N_rad)
    #
    # Form a pixel mask for each search zone, and assemble the masks into two tables (1-D and 2-D formats).
    #
    ref_table_L, zonemask_table_1d_L, zonemask_table_2d_L = get_ref_and_pix_tables(xycent=None, fr_shape=fr_shape, N_fr=N_fr_L,
                                                                                   op_fr=op_fr_L, N_rad=N_rad, R_inner=R_inner, R_out=R_out,
                                                                                   op_rad=op_rad, N_az=N_az, op_az=op_az,
                                                                                   parang_seq=parang_seq_L, fwhm=fwhm,
                                                                                   min_refgap_fac=min_refgap_fac, track_mode=track_mode,
                                                                                   diagnos_stride=diagnos_stride)
    # 
    # Perform zone-by-zone KLIP subtraction on each frame
    #
    klip_config_L = {'fr_shape':fr_shape, 'parang_seq':parang_seq_L, 'mode_cut':mode_cut,
                     'track_mode':track_mode, 'op_fr':op_fr_L, 'op_rad':op_rad, 'op_az':op_az,
                     'ref_table':ref_table_L, 'zonemask_table_1d':zonemask_table_1d_L,
                     'zonemask_table_2d':zonemask_table_2d_L}
    klip_data_L = [[[dict.fromkeys(['I', 'I_mean', 'Z', 'sv', 'Projmat', 'I_proj', 'F']) for a in range(N_az[r])] for r in range(N_rad)] for i in range(N_fr_L)]
    klipsub_cube_L, klippsf_cube_L, derot_klipsub_cube_L = do_klip_subtraction(data_cube=data_cube_L, config_dict=klip_config_L,
                                                                               result_dict=klip_data_L, result_dir=result_dir,
                                                                               diagnos_stride=diagnos_stride, store_klbasis=True)
    #
    # Coadd the derotated, subtracted images and organize the results
    #
    coadd_img_L = np.mean(derot_klipsub_cube_L, axis=0)
    med_img_L = np.median(derot_klipsub_cube_L, axis=0)
    mean_klippsf_img_L = np.mean(klippsf_cube_L, axis=0)
    med_klippsf_img_L = np.median(klippsf_cube_L, axis=0)
    #
    # Get statistics from co-added and median residual images
    #
    annular_rms, zonal_rms = get_residual_stats(config_dict=klip_config_L, Phi_0=Phi_0,
                                                coadd_img=coadd_img_L, med_img=med_img_L)
    if store_results == True:
        delimiter = '-'
        label_str = "globalklip_rad%s_mode%s" % (delimiter.join(["%02d" % r for r in R_out]), delimiter.join(["%02d" % m for m in mode_cut]))
        klipsub_cube_L_fname = "%s/%s_res_cube_L.fits" % (result_dir, label_str)
        klippsf_cube_L_fname = "%s/%s_psf_cube_L.fits" % (result_dir, label_str)
        derot_klipsub_cube_L_fname = "%s/%s_derot_res_cube_L.fits" % (result_dir, label_str)
        coadd_img_L_fname = "%s/%s_res_coadd_L.fits" % (result_dir, label_str)
        med_img_L_fname = "%s/%s_res_med_L.fits" % (result_dir, label_str)
        mean_klippsf_img_L_fname = "%s/%s_psf_mean_L.fits" % (result_dir, label_str)
        med_klippsf_img_L_fname = "%s/%s_psf_med_L.fits" % (result_dir, label_str)
        klipsub_archv_fname = "%s/%s_klipsub_archv.shelve" % (result_dir, label_str)

        klippsf_cube_L_hdu = pyfits.PrimaryHDU(klippsf_cube_L.astype(np.float32))
        klippsf_cube_L_hdu.writeto(klippsf_cube_L_fname, clobber=True)
        print "\nWrote KLIP PSF estimate cube (%.3f Mb) to %s" % (klippsf_cube_L.nbytes/10.**6, klippsf_cube_L_fname)
        mean_klippsf_img_L_hdu = pyfits.PrimaryHDU(mean_klippsf_img_L.astype(np.float32))
        mean_klippsf_img_L_hdu.writeto(mean_klippsf_img_L_fname, clobber=True)
        print "Wrote average of KLIP PSF estimate cube (%.3f Mb) to %s" % (mean_klippsf_img_L.nbytes/10.**6, mean_klippsf_img_L_fname)
        med_klippsf_img_L_hdu = pyfits.PrimaryHDU(med_klippsf_img_L.astype(np.float32))
        med_klippsf_img_L_hdu.writeto(med_klippsf_img_L_fname, clobber=True)
        print "Wrote median of KLIP PSF estimate cube (%.3f Mb) to %s" % (med_klippsf_img_L.nbytes/10.**6, med_klippsf_img_L_fname)
        klipsub_cube_hdu = pyfits.PrimaryHDU(klipsub_cube_L.astype(np.float32))
        klipsub_cube_hdu.writeto(klipsub_cube_L_fname, clobber=True)
        print "Wrote KLIP-subtracted cube (%.3f Mb) to %s" % (klipsub_cube_L.nbytes/10.**6, klipsub_cube_L_fname)
        derot_klipsub_cube_L_hdu = pyfits.PrimaryHDU(derot_klipsub_cube_L.astype(np.float32))
        derot_klipsub_cube_L_hdu.writeto(derot_klipsub_cube_L_fname, clobber=True)
        print "Wrote derotated, KLIP-subtracted image cube (%.3f Mb) to %s" % (derot_klipsub_cube_L.nbytes/10.**6, derot_klipsub_cube_L_fname)
        coadd_img_L_hdu = pyfits.PrimaryHDU(coadd_img_L.astype(np.float32))
        coadd_img_L_hdu.writeto(coadd_img_L_fname, clobber=True)
        print "Wrote average of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (coadd_img_L.nbytes/10.**6, coadd_img_L_fname)
        med_img_L_hdu = pyfits.PrimaryHDU(med_img_L.astype(np.float32))
        med_img_L_hdu.writeto(med_img_L_fname, clobber=True)
        print "Wrote median of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (med_img_L.nbytes/10.**6, med_img_L_fname)
        if os.path.exists(klipsub_archv_fname):
            os.remove(klipsub_archv_fname)
        if store_archv:
            klipsub_archv = shelve.open(klipsub_archv_fname)
            klipsub_archv['klip_config_L'] = klip_config_L
            klipsub_archv['klip_data_L'] = klip_data_L
            klipsub_archv.close()
            print "Wrote KLIP reduction (%.3f Mb) archive to %s" % (os.stat(klipsub_archv_fname).st_size/10.**6, klipsub_archv_fname)
