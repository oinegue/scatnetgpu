%% build_filters_for_python: function description
function [filters] = build_filters_for_python_fft(J,L,shape,margin)

	addpath_scatnet;

	filt_opt = struct();
	filt_opt.J = J;
	filt_opt.L = L;
	filt_opt.filter_type='morlet';

	scat_opt = struct();

	[Wop2, filters] = wavelet_factory_2d(shape,filt_opt, scat_opt);