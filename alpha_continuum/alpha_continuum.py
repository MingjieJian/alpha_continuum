import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import norm
from scipy.signal import savgol_filter, correlate
from scipy.interpolate import UnivariateSpline, Akima1DInterpolator, interp1d

# Remove the peaks
def peak_removal(spec, n_iter=5, printout=False, plot=False, plot_save_dir=None, max_roll_width=100, quantile_width=5):

    dwave = np.mean(np.diff(spec['wave']))
    flux = spec['flux'].values.copy()
    mask_overall = (flux > 0) & (flux < 0)
    
    for iteration in range(n_iter): 
        # sigma clipping on 100 \AA range, 99%
        maxi_roll_fast = np.ravel(pd.DataFrame(flux).rolling(int(max_roll_width/dwave), min_periods=1, center=True).quantile(0.99))
        Q3_fast = np.ravel(pd.DataFrame(flux).rolling(int(quantile_width/dwave), min_periods=1, center=True).quantile(0.75))  
        Q2_fast = np.ravel(pd.DataFrame(flux).rolling(int(quantile_width/dwave), min_periods=1, center=True).quantile(0.50))  
        Q1_fast = np.ravel(pd.DataFrame(flux).rolling(int(quantile_width/dwave), min_periods=1, center=True).quantile(0.25))
        IQ_fast = 3 * (Q3_fast-Q2_fast)
        sup_fast = Q3_fast + IQ_fast
        
        if printout:
            print(' Number of cosmic peaks removed : %.0f'%(np.sum((flux > sup_fast) & (flux > maxi_roll_fast))))
        
        mask = (flux > sup_fast) & (flux > maxi_roll_fast)
        mask_overall = mask_overall | mask

        if sum(mask)==0:
            break
        flux[mask] = Q2_fast[mask]

    spec['flux_peaks_removed'] = flux
    spec['indices_peaks'] = mask_overall
    spec['flux_peaks_removed_smoothed'] = np.ravel(pd.DataFrame(spec['flux_peaks_removed']).rolling(int(3/dwave), min_periods=1, center=True).quantile(0.50))

    if plot:
        plt.figure(figsize=(13, 3), dpi=150)
        plt.plot(spec['wave'], spec['flux'], lw=0.5, label='flux')
        plt.plot(spec['wave'], spec['flux_peaks_removed'], lw=0.5, label='flux_peaks_removed')
        plt.plot(spec.loc[mask_overall, 'wave'], spec.loc[mask_overall, 'flux'], 
                 'x', color='red', markersize=4, alpha=0.5, label='peaks removed')
        plt.xlabel('Wavelength ($\mathrm{\AA}$)')
        plt.legend()
        if plot_save_dir is not None:
            plt.savefig(f'{plot_save_dir}/peak_removal.pdf')
        
    return spec

def snr_smooth(spec, max_smooth_width=10, plot=False, plot_save_dir=None):

    dwave = np.mean(np.diff(spec['wave']))
    
    if 'snr' not in spec.columns:
        spec['snr'] = np.sqrt(spec['flux'])
        spec['snr_con'] = rough_continuum(np.sqrt(spec['flux']), np.mean(np.diff(spec['wave'])), quantile=0.8)
    else:
        spec['snr_con'] = rough_continuum(spec['snr'], np.mean(np.diff(spec['wave'])), quantile=0.8)

    snr_smooth_width = np.ceil((100 / spec['snr_con'])**2)
    snr_smooth_width[snr_smooth_width < 1] = 1
    snr_smooth_width[snr_smooth_width > max_smooth_width] = max_smooth_width
    spec['snr_smooth_width'] = snr_smooth_width
    spec['flux_snr_smooth'] = spec['flux'].copy()
    # print(spec.groupby('snr_smooth_width').size().index)
    if not (spec['snr_smooth_width'] == 1).all():
        for width in spec.groupby('snr_smooth_width').size().index:
            indices = spec['snr_smooth_width'] == width
            spec.loc[indices, 'flux_snr_smooth'] = np.float32(np.ravel(spec.loc[indices, 'flux_snr_smooth'].rolling(int(width/dwave), min_periods=1, center=True).quantile(0.5)))

    if plot:
        plt.figure(figsize=(13, 3), dpi=150)
        plt.plot(spec['wave'], spec['flux'], lw=0.5, label='flux')
        plt.plot(spec['wave'], spec['flux_snr_smooth'], lw=0.5, label='flux_snr_smooth')
        plt.xlabel('Wavelength ($\mathrm{\AA}$)')
        plt.legend()
        if plot_save_dir is not None:
            plt.savefig(f'{plot_save_dir}/snr_smooth.pdf')
    
    return spec

# Determine the alpha radius 
def smooth(y, box_pts, shape='rectangular'): #rectangular kernel for the smoothing
    box2_pts = int(2*box_pts-1)
    if shape=='savgol':
        if box2_pts>=5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    else:
        if shape=='rectangular':
            box = np.ones(box2_pts)/box2_pts
        if shape == 'gaussian':
            vec = np.arange(-25,26)
            box = norm.pdf(vec,scale=(box2_pts-0.99)/2.35)/np.sum(norm.pdf(vec,scale = (box2_pts-0.99)/2.35))
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth[0:int((len(box)-1)/2)] = y[0:int((len(box)-1)/2)]
        y_smooth[-int((len(box)-1)/2):] = y[-int((len(box)-1)/2):]
    return y_smooth    

def ccf_cal(wave, spec1, spec2, extended=1500):
    """
    CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask, mask_telluric = binary mask
    
    Parameters
    ----------
    extended : int-type 
    
    Returns
    -------
    
    Return the vrad lag grid as well as the CCF
    
    """
       
    dwave = wave[1]-wave[0]
    spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])
    spec2 = np.hstack([np.zeros(extended),spec2,np.zeros(extended)])
    wave = np.hstack([np.arange(-extended*dwave+wave.min(),wave.min(),dwave),wave,np.arange(wave.max()+dwave,(extended+1)*dwave+wave.max(),dwave)])
    shift = np.linspace(0,dwave,10)[:-1]
    shift_save = []
    sum_spec = np.sum(spec2)
    convolution = []
    for j in shift:
        new_spec = interp1d(wave+j,spec2,kind='cubic', bounds_error=False, fill_value='extrapolate')(wave)
        for k in np.arange(-60,61,1):
            new_spec2 = np.hstack([new_spec[-k:],new_spec[:-k]])
            convolution.append(np.sum(new_spec2*spec1)/sum_spec)
            shift_save.append(j+k*dwave)
    shift_save = np.array(shift_save)
    return (299.792e6*10**shift_save)-299.792e6, np.array(convolution)
    
def gaussian(x, cen, amp, offset, wid):
    return amp * np.exp(-0.5*(x-cen)**2 / wid**2)+offset

def rough_continuum(flux, dwave, rollmax_width=30, smooth_width=15, quantile=1):
    flux_use = flux.copy()
    flux_use[flux_use < 0.01] = 0.01
    
    # Rolling maxima in a 30 angstrom window, left to right
    continuum_right = np.ravel(pd.DataFrame(flux_use).rolling(int(rollmax_width/dwave)).quantile(quantile)) 
    # Rolling maxima in a 30 angstrom window, right to left
    continuum_left = np.ravel(pd.DataFrame(flux_use[::-1]).rolling(int(rollmax_width/dwave)).quantile(quantile))[::-1]
    continuum_right[np.isnan(continuum_right)] = continuum_right[~np.isnan(continuum_right)][0] 
    continuum_left[np.isnan(continuum_left)] = continuum_left[~np.isnan(continuum_left)][-1]
    both = np.array([continuum_right, continuum_left])
    continuum = np.min(both,axis=0)
    # Smoothing of the envelop 15 anstromg to provide more accurate weight
    continuum = smooth(continuum, int(smooth_width/dwave), shape='rectangular') 
    return continuum

def log_wav2rv(wav):
    '''
    Convert the log wavelangth scale to RV.

    :param wav:
        A numpy.array, containing the logarithmic wavelength scale of the spectra.
    :return:
        The RV array. 
    '''
    
    # check if the a value is constant for all the pixels
    if not np.all(wav):
        raise ValueError('The wavelength array is not in constant log scale.')
        
    c = 2.99792e5
    a = wav[1] / wav[0]
    if len(wav) % 2 == 1:
        # Odd case
        multiplicative_shift = a**(np.arange(1, (len(wav)-1) // 2 + 1) / 1)
        rv_array = c * (a**np.arange(1, len(wav) // 2 + 1) - 1)
        rv_array = c * (multiplicative_shift**2 - 1) / (multiplicative_shift**2 + 1)
        rv_array = np.concatenate([-rv_array[::-1], [0], rv_array])
    else:
        # Even case
        wav_temp = np.append(wav, wav[0] * a**len(wav))
        rv_array = log_wav2rv(wav_temp)[:-1]
            
    return rv_array

def determine_line_width(spec, printout=False, plot=False, plot_save_dir=None):
    
    dwave = np.mean(np.diff(spec['wave']))
    mask = np.zeros(len(spec))

    continuum = rough_continuum(spec['flux_peaks_removed_smoothed'], dwave)

    # Place the raw-continuum normalized spectrum in log wavelength scale. 
    log_grid = np.linspace(np.log10(spec['wave']).min(), np.log10(spec['wave']).max(), len(spec['wave']))
    log_spectrum = interp1d(np.log10(spec['wave']), spec['flux_peaks_removed_smoothed']/continuum, kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid)
    
    ccf = correlate((log_spectrum - np.mean(log_spectrum)) / np.std(log_spectrum), (log_spectrum - np.mean(log_spectrum)) / np.std(log_spectrum), mode='same') / len(log_spectrum)
    # ccf_pixel = np.arange(len(ccf))
    # ccf_pixel -= ccf_pixel[np.argmax(ccf)]
    ccf_rv = log_wav2rv(10**log_grid)

    fwhm = ccf_rv[ccf > 0.5]
    # return fwhm
    fwhm_km = fwhm[-1] - fwhm[0]
    
    if plot:
        plt.figure(figsize=(13, 3), dpi=150)
        plt.plot(ccf_rv, ccf)
        plt.plot([-fwhm_km/2, fwhm_km/2], [0.5, 0.5])
        plt.xlim(-10*fwhm_km, 10*fwhm_km)
        plt.xlabel('Radial Velocity (km/s)')
        plt.ylabel('CCF')
        plt.title(f' FWHM computed from the CCF is about: {fwhm_km:.2f} [km/s]')
        if plot_save_dir is not None:
            plt.savefig(f'{plot_save_dir}/line_width.pdf')
    
    if printout:
        print(f' [AUTO] FWHM computed from the CCF is about : {fwhm_km:.2f} [km/s]')

    return fwhm_km

def determine_alpha_radius(spec, line_fwhm, base_ratio=2, penality_ratio=1, plot=False, plot_save_dir=None):
    # Calculate the alpha-radius
    continuum_large = rough_continuum(spec['flux_peaks_removed'], np.mean(np.diff(spec['wave'])), rollmax_width=30, quantile=0.9)
    penalite0 = (continuum_large - spec['flux_peaks_removed'])/continuum_large
    penalite0 /= np.max(penalite0)
    r = line_fwhm/3e5*spec['wave']
    r *= base_ratio
    penality = r * (1 + penalite0*penality_ratio)
    penality_step = np.round(penality, decimals=0)
    penality_step[penality_step<line_fwhm/3e5*spec['wave']] = line_fwhm/3e5*spec.loc[penality_step<line_fwhm/3e5*spec['wave'], 'wave']
    spec['radius'] = penality_step

    if plot:
        plt.figure(figsize=(13, 3), dpi=150)
        line1, = plt.plot(spec['wave'], spec['flux_peaks_removed'], c='gray', lw=0.5, label='flux_peaks_removed (left)')
        line2, = plt.plot(spec['wave'], continuum_large, lw=1, label='Rough continuum (left)', c='C1')
        
        plt.twinx()
        line3, = plt.plot(spec['wave'], penality_step, lw=0.5, c='C3', label='alpha radius (right)')
        lines = [line1, line2, line3]
        plt.legend(lines, [line.get_label() for line in lines])
        plt.xlabel('Wavelength ($\mathrm{\AA}$)')
        if plot_save_dir is not None:
            plt.savefig(f'{plot_save_dir}/alpha_radius.pdf')
            
    return spec

def find_next_edge_point(spec, flux, radius=-1):
    start_i = spec.index[0]
    if radius == -1:
        radius = spec.loc[start_i, 'radius']
    
    # We start from the left end.
    # Find the next edge point to the most right one.
    pixel_x = spec.loc[start_i, 'wave']
    pixel_y = spec.loc[start_i, flux]
    # Find all pixels on the right side with in the radius
    distance = np.sqrt((spec['wave'] - pixel_x)**2 + (spec[flux] - pixel_y)**2)
    distance *= np.sign(spec['wave'] - pixel_x)
    
    spec_chunk = spec[(distance <= radius*2) & (distance > 0)]

    # Find the ledder point, from the most right hand side
    edge_length = len(spec_chunk)
    if edge_length == 0:
        # The first pixel is away from the other pixels. Return to find_all_edge_points and start from the next pixel.
        return -99
    iter_N = 0
    while edge_length > 0:
        pixel_x_next, pixel_y_next = spec_chunk.iloc[-1]['wave'], spec_chunk.iloc[-1][flux]
        pixel_index_next = spec_chunk.index[-1]
        line_paras = np.polyfit([pixel_x, pixel_x_next], [pixel_y, pixel_y_next], 1)
        indices = spec_chunk[flux] > np.polyval(line_paras, spec_chunk['wave']) + 1
        # plt.scatter(spec_chunk.loc[indices, 'wave'], spec_chunk.loc[indices, 'flux_stretch'], s=1)
        spec_chunk = spec_chunk[indices]
        
        if edge_length == len(spec_chunk):
            print('Iteration unchanged')
            break
        edge_length = len(spec_chunk)
        iter_N += 1
    
    return pixel_index_next

def find_all_edge_points(spec, flux, radius=-1):
    spec_use = spec.copy()
    spec_out = spec.copy()
    spec_out.loc[spec_out.index[0], 'edge'] = True

    count = 1
    while len(spec_use) > 1:
        pixel_index_next = find_next_edge_point(spec_use, flux, radius)
        while pixel_index_next == -99:
            spec_out.loc[spec_use.index[0], 'edge'] = True
            spec_use = spec_use.iloc[1:]
            if len(spec_use) == 0:
                break
            pixel_index_next = find_next_edge_point(spec_use, flux, radius)

        if pixel_index_next != -99:
            spec_out.loc[pixel_index_next, 'edge'] = True
            spec_use = spec_use.loc[pixel_index_next:]
        count += 1
        # print(len(spec_use))

    return spec_out

def rolling_line(spec, stretch=True, fit_method='poly', plot=False, plot_save_dir=None, poly_deg=8):
    if stretch:
        stretch_ratio = np.ptp(spec['flux']) / np.ptp(spec['wave']) # stretch the y axis to scale the x and y axis
    else:
        stretch_ratio = 1
    spec['flux_peaks_removed_smoothed_stretch'] = spec['flux_peaks_removed_smoothed'] / stretch_ratio
    spec['flux_stretch'] = spec['flux'] / stretch_ratio  
    spec['edge'] = False

    # Roll the line
    spec = find_all_edge_points(spec, 'flux_peaks_removed_smoothed_stretch')
    
    if fit_method == 'poly':
        poly_fitting = np.polyfit(spec.loc[spec['edge'], 'wave'], spec.loc[spec['edge'], 'flux_peaks_removed_smoothed_stretch'], poly_deg)
        spec['flux_stretch_normed'] = spec['flux_stretch'] / np.polyval(poly_fitting, spec['wave'])
        spec['continuum'] = np.polyval(poly_fitting, spec['wave'])
    elif fit_method == 'spline':
        cs = UnivariateSpline(spec.loc[spec['edge'], 'wave'], spec.loc[spec['edge'], 'flux_peaks_removed_smoothed_stretch'])
        spec['flux_stretch_normed'] = spec['flux_stretch'] / cs(spec['wave'])
        spec['continuum'] = cs(spec['wave'])
    elif fit_method == 'akima':
        cs = Akima1DInterpolator(spec.loc[spec['edge'], 'wave'], spec.loc[spec['edge'], 'flux_peaks_removed_smoothed_stretch'])
        spec['flux_stretch_normed'] = spec['flux_stretch'] / cs(spec['wave'])
        spec['continuum'] = cs(spec['wave'])
    else:
        raise ValueError('The interpoation method is not supported.') 

    if plot:
        plt.figure(figsize=(13, 3), dpi=150)
        plt.plot(spec.loc[:, 'wave'], spec.loc[:, 'flux_stretch'], lw=0.5, zorder=0, label='flux')
        plt.scatter(spec.loc[spec['edge'], 'wave'], spec.loc[spec['edge'], 'flux_peaks_removed_smoothed_stretch'], 
                    s=1, color='red', label='continuum points')
        plt.plot(spec['wave'], spec['continuum'], lw=1, label='continuum')
        plt.legend()
        plt.xlabel('Wavelength ($\mathrm{\AA}$)')
        if plot_save_dir is not None:
            plt.savefig(f'{plot_save_dir}/rolling_line.pdf')

    return spec

def normalization(spec, stretch=True, fit_method='poly', base_ratio=2, penality_ratio=1, poly_deg=8, printout=False, plot=False, plot_save_dir=None):
    '''
    The main function to perform normalization.
    '''
    spec = peak_removal(spec.copy(), printout=printout, plot=plot, plot_save_dir=plot_save_dir)
    spec = snr_smooth(spec, plot=plot, plot_save_dir=plot_save_dir)
    line_fwhm = determine_line_width(spec, printout=printout, plot=plot, plot_save_dir=plot_save_dir)
    spec = determine_alpha_radius(spec, line_fwhm, base_ratio=base_ratio, penality_ratio=penality_ratio, plot=plot, plot_save_dir=plot_save_dir)
    spec = rolling_line(spec, stretch=stretch, fit_method=fit_method, plot=plot, plot_save_dir=plot_save_dir, poly_deg=poly_deg)
    return spec