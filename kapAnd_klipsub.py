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
import sys
import os
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
            reconstrd_cube[fr_ind, row, col] = pix_val
    return reconstrd_cube

def reconst_zone(data_vec, pix_table, img_dim):
    reconstrd_img = np.zeros(img_dim)
    for i, pix_val in enumerate(data_vec.flat):
        row = pix_table[0][i]
        col = pix_table[1][i]
        reconstrd_img[row, col] = pix_val
    return reconstrd_img

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

def get_pca_basis(R, cutoff):
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[0:cutoff, :], sv

def get_klip_basis(R, cutoff):
    w, V = np.linalg.eig(np.dot(R, np.transpose(R)))
    sort_ind = np.argsort(w)[::-1] #indices of eigenvals sorted in descending order
    sv = np.sqrt(w[sort_ind]).reshape(-1,1) #column of ranked singular values
    Z = np.dot(1./sv*np.transpose(V[:, sort_ind]), R)
    return Z[0:cutoff, :], sv

if __name__ == "__main__":
    #
    # Set KLIP parameters
    #
    klip_mode = True 
    mean_sub = True

    #
    # point PCA
    #
    track_mode = True
    mode_cut = [10]
    R_inner = 85.
    R_out = [110.]
    DPhi = [30.]
    #DPhi = [20.]
    #DPhi = [72.]
    Phi_0 = [328.]
    #Phi_0 = [-6.]
    #
    # global PCA
    #
    #track_mode = False
    #mode_cut = [10, 10]
    #R_inner = 20.
    #R_out = [60., 80.]
    #DPhi = [360., 360.]
    #Phi_0 = [0., 0.]

    N_rad = len(R_out)
    fwhm = 11.
    min_refgap_fac = 1.5
    assert(len(mode_cut) == N_rad == len(DPhi) == len(Phi_0))
    N_az = [ int(np.ceil(360./DPhi[r])) for r in range(N_rad) ]
    #
    # Load data
    #
    data_dir = os.path.expanduser('~/Data/LMIRcam/kappaAnd')
    result_dir = os.path.expanduser('~/Data/LMIRcam/kappaAnd/klipsub_results')
    assert(os.path.exists(data_dir)), 'data_dir %s does not exist' % data_dir
    assert(os.path.exists(result_dir)), 'result_dir %s does not exist' % result_dir
    cube_fname = "%s/cube_Beth.fits" % data_dir 
    #cube_fname = "%s/cube_fakeplanets_Neil.fits" % data_dir 
    parang_fname = "%s/parang_Beth.fits" % data_dir
    #Data_cube = load_data_cube(cube_fname, R_out[-1])
    #Data_cube = load_and_clean_data_cube(cube_fname, R_out[-1], replace_bad = False)
    Data_cube = load_and_clean_data_cube(cube_fname, R_out[-1], replace_bad = True)
    parang_hdu = pyfits.open(parang_fname, 'readonly')
    parang_seq = parang_hdu[0].data.copy()
    parang_hdu.close()
    N_fr = Data_cube.shape[0]
    fr_shape = Data_cube.shape[1:]
    fr_width = fr_shape[1]
    N_parang = parang_seq.shape[0]
    assert(np.equal(N_fr, N_parang))
    print "Loaded kappa And cube contains %d images of width %d pixels." % (N_fr, fr_width)
    print "Corresponding parallactic angle array has %d entries ranging from %0.2f to %0.2f deg." %\
          (N_parang, parang_seq[0], parang_seq[N_parang-1])
    #
    # Set additional program parameters
    #
    store_results = True
    #store_results = False
    store_archv = True
    #store_archv = False
    diagnos_stride = 12
    op_fr = np.arange(N_fr)
    op_rad = range(N_rad)
    N_op_fr = op_fr.shape[0]
    #op_az = [range(N_az[i]) for i in range(N_rad)]
    #op_az = [[0, 6]]
    op_az = [[0]]
    #op_fr = np.array([12])
    #op_fr = np.arange(0, N_fr, diagnos_stride)
    assert(len(op_rad) == len(op_az) == N_rad)
    #
    # Form a pixel mask for each search zone, and assemble the masks into two tables (1-D and 2-D formats).
    #
    x_cntr = fr_width/2 - 0.5 #zero-based index of x-center
    y_cntr = fr_width/2 - 0.5 #zero-based index of y-center
    rad_vec = np.sqrt(get_radius_sqrd(fr_shape, (x_cntr, y_cntr))).ravel()
    angle_vec = get_angle(fr_shape, (x_cntr, y_cntr)).ravel()
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
                min_refang = np.arctan(min_refgap_fac*fwhm/((R1 + R2)/2))*180/np.pi
            ref_table[fr_ind][rad_ind] = np.where(np.greater_equal(np.abs(parang_seq - parang_seq[fr_ind]), min_refang))[0]
            if fr_ind%diagnos_stride == 0:
                print "\tFrame %d/%d, annulus %d/%d: %d valid reference frames." %\
                      (fr_ind+1, N_fr, rad_ind+1, N_rad, len(ref_table[fr_ind][rad_ind]))
            assert(len(ref_table[fr_ind][rad_ind]) > 0)
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
    print "Search zone scheme:"
    if track_mode:
        print "\tTrack mode ON"
    else:
        print "\tTrack mode OFF"
    print "\tR_inner:", R_inner, "; R_out:", R_out
    print "\tPhi_0, DPhi, N_az:", Phi_0, DPhi, N_az
    print "\tmode_cut:", mode_cut

    # 
    # Perform zone-by-zone KLIP subtraction on each frame
    #
    klipsub_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    klippsf_cube = np.zeros((N_op_fr, fr_shape[0], fr_shape[1]))
    derot_klipsub_cube = klipsub_cube.copy()
    klip_config = {'fr_shape':fr_shape, 'parang_seq':parang_seq, 'mode_cut':mode_cut,\
                   'track_mode':track_mode, 'op_fr':op_fr, 'op_rad':op_rad, 'op_az':op_az,\
                   'ref_table':ref_table, 'zonemask_table_1d':zonemask_table_1d, 'zonemask_table_2d':zonemask_table_2d}
    klip_data = [[[dict.fromkeys(['I', 'I_mean', 'Z', 'sv', 'Projmat', 'I_proj', 'F']) for a in range(N_az[r])] for r in range(N_rad)] for i in range(N_fr)]
    start_time = time.time()
    for i, fr_ind in enumerate(op_fr):
        # Loop over operand frames
        if fr_ind%diagnos_stride == 0:
            klbasis_cube = np.zeros((np.amin(mode_cut), fr_shape[0], fr_shape[1]))
        for rad_ind in op_rad:
            for az_ind in op_az[rad_ind]:
                I = np.ravel(Data_cube[fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy() 
                R = np.zeros((ref_table[fr_ind][rad_ind].shape[0], zonemask_table_1d[fr_ind][rad_ind][az_ind].shape[0]))
                for j, ref_fr_ind in enumerate(ref_table[fr_ind][rad_ind]):
                    R[j,:] = np.ravel(Data_cube[ref_fr_ind,:,:])[ zonemask_table_1d[fr_ind][rad_ind][az_ind] ].copy()
                if klip_mode == True: # following Soummer et al. 2012
                    if mean_sub == True:
                        I_mean = np.mean(I)
                        I -= I_mean
                        R -= R.mean(axis=1).reshape(-1, 1)
                    Z, sv = get_klip_basis(R = R, cutoff = mode_cut[rad_ind])
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
                    klbasis_cube += reconst_zone_cube(Z, zonemask_table_2d[fr_ind][rad_ind][az_ind], klbasis_cube.shape)
                    if mean_sub == False:
                        I_mean = 0
                    print "Frame %d/%d, annulus %d/%d, sector %d/%d: RMS before/after subtraction: %0.2f / %0.2f" %\
                          (fr_ind+1, N_fr, rad_ind+1, N_rad, az_ind+1, N_az[rad_ind],\
                           np.sqrt(np.mean((I + I_mean)**2)), np.sqrt(np.mean(F**2)))
                    #if rad_ind == 0 and az_ind == 0:
                    #    reconst_search_zone = reconst_zone(I, zonemask_table_2d[rad_ind][az_ind], fr_shape)
                    #    reconst_est_zone = reconst_zone(I_est, zonemask_table_2d[rad_ind][az_ind], fr_shape)
                    #    #plt.subplot(121)
                    #    #plt.imshow(reconst_search_zone, origin='lower', interpolation='nearest', norm=matplotlib.colors.LogNorm(), cmap='gray')
                    #    #plt.subplot(122)
                    #    #plt.imshow(reconst_est_zone, origin='lower', interpolation='nearest', norm=matplotlib.colors.LogNorm(), cmap='gray')
                    #    #plt.show()
                    #    search_hdu = pyfits.PrimaryHDU(reconst_search_zone)
                    #    est_hdu = pyfits.PrimaryHDU(reconst_est_zone)
                    #    search_hdu.writeto("%s/test_search_zone_fr%02d.fits" % (result_dir, fr_ind), clobber=True)
                    #    est_hdu.writeto("%s/test_est_zone_fr%02d.fits" % (result_dir, fr_ind), clobber=True)
        # De-rotate the KLIP-subtracted image
        derot_klipsub_img = rotate(klipsub_cube[i,:,:], -parang_seq[fr_ind], reshape=False)
        derot_klipsub_cube[i,:,:] = derot_klipsub_img
        if fr_ind % diagnos_stride == 0:
            print "***** Frame %d/%d has been PSF-sub'd and derotated. *****" % (fr_ind+1, N_fr)
            if store_results == True:
                klbasis_cube_hdu = pyfits.PrimaryHDU(klbasis_cube.astype(np.float32))
                klbasis_cube_hdu.writeto("%s/kapAnd_klbasis_fr%02d.fits" % (result_dir, fr_ind), clobber=True)
    #
    # Coadd the derotated, subtracted images and organize the results
    #
    coadd_img = np.mean(derot_klipsub_cube, axis=0)
    med_img = np.median(derot_klipsub_cube, axis=0)
    mean_klippsf_img = np.mean(klippsf_cube, axis=0)
    med_klippsf_img = np.median(klippsf_cube, axis=0)
    end_time = time.time()
    exec_time = end_time - start_time
    time_per_frame = exec_time/N_op_fr
    print "Took %dm%02ds to KLIP-subtract %d frames (%0.2f s per frame)." %\
          (int(exec_time/60.), exec_time - 60*int(exec_time/60.), N_op_fr, time_per_frame)

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
        print "\tannulus %d/%d: %.2f in KLIP sub'd, derotated, coadded annlus" % (rad_ind+1, N_rad, coadd_annular_rms[-1])
        if len(op_az[rad_ind]) > 1:
            Phi_beg = (Phi_0_derot - DPhi[rad_ind]/2.) % 360.
            Phi_end = [ (Phi_beg + i * DPhi[rad_ind]) % 360. for i in range(1, N_az[rad_ind]) ]
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

    if store_results == True:
        delimiter = '_'
        label_str = "cut%s_delPhi%s" % (delimiter.join(["%02d" % m for m in mode_cut]), delimiter.join(["%02d" % dphi for dphi in DPhi]))
        klipsub_cube_fname = "%s/kapAnd_%s_klipsub_cube.fits" % (result_dir, label_str)
        klippsf_cube_fname = "%s/kapAnd_%s_klippsf_cube.fits" % (result_dir, label_str)
        derot_klipsub_cube_fname = "%s/kapAnd_%s_derot_klipsub_cube.fits" % (result_dir, label_str)
        coadd_img_fname = "%s/kapAnd_%s_coadd.fits" % (result_dir, label_str)
        med_img_fname = "%s/kapAnd_%s_med.fits" % (result_dir, label_str)
        mean_klippsf_img_fname = "%s/kapAnd_%s_klippsf_mean.fits" % (result_dir, label_str)
        med_klippsf_img_fname = "%s/kapAnd_%s_klippsf_med.fits" % (result_dir, label_str)
        klipsub_archv_fname = "%s/kapAnd_%s_klipsub_archive.shelve" % (result_dir, label_str)

        klippsf_cube_hdu = pyfits.PrimaryHDU(klippsf_cube.astype(np.float32))
        klippsf_cube_hdu.writeto(klippsf_cube_fname, clobber=True)
        print "Wrote KLIP PSF estimate cube (%.3f Mb) to %s" % (klippsf_cube.nbytes/10.**6, klippsf_cube_fname)
        mean_klippsf_img_hdu = pyfits.PrimaryHDU(mean_klippsf_img.astype(np.float32))
        mean_klippsf_img_hdu.writeto(mean_klippsf_img_fname, clobber=True)
        print "Wrote average of KLIP PSF estimate cube (%.3f Mb) to %s" % (mean_klippsf_img.nbytes/10.**6, mean_klippsf_img_fname)
        med_klippsf_img_hdu = pyfits.PrimaryHDU(med_klippsf_img.astype(np.float32))
        med_klippsf_img_hdu.writeto(med_klippsf_img_fname, clobber=True)
        print "Wrote median of KLIP PSF estimate cube (%.3f Mb) to %s" % (med_klippsf_img.nbytes/10.**6, med_klippsf_img_fname)
        klipsub_cube_hdu = pyfits.PrimaryHDU(klipsub_cube.astype(np.float32))
        klipsub_cube_hdu.writeto(klipsub_cube_fname, clobber=True)
        print "Wrote KLIP-subtracted cube (%.3f Mb) to %s" % (klipsub_cube.nbytes/10.**6, klipsub_cube_fname)
        derot_klipsub_cube_hdu = pyfits.PrimaryHDU(derot_klipsub_cube.astype(np.float32))
        derot_klipsub_cube_hdu.writeto(derot_klipsub_cube_fname, clobber=True)
        print "Wrote derotated, KLIP-subtracted image cube (%.3f Mb) to %s" % (derot_klipsub_cube.nbytes/10.**6, derot_klipsub_cube_fname)
        coadd_img_hdu = pyfits.PrimaryHDU(coadd_img.astype(np.float32))
        coadd_img_hdu.writeto(coadd_img_fname, clobber=True)
        print "Wrote average of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (coadd_img.nbytes/10.**6, coadd_img_fname)
        med_img_hdu = pyfits.PrimaryHDU(med_img.astype(np.float32))
        med_img_hdu.writeto(med_img_fname, clobber=True)
        print "Wrote median of derotated, KLIP-subtracted images (%.3f Mb) to %s" % (med_img.nbytes/10.**6, med_img_fname)
        if os.path.exists(klipsub_archv_fname):
            os.remove(klipsub_archv_fname)
        if store_archv:
            klipsub_archv = shelve.open(klipsub_archv_fname)
            klipsub_archv['klip_config'] = klip_config
            klipsub_archv['klip_data'] = klip_data
            klipsub_archv.close()
            print "Wrote KLIP reduction (%.3f Mb) archive to %s" % (os.stat(klipsub_archv_fname).st_size/10.**6, klipsub_archv_fname)
