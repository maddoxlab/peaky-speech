form Test command line calls
	sentence file_path_no_ext file_name
	real f_min 75.0
	real f_max 600.0
endform
Read from file: file_path_no_ext$ + ".wav"
To Pitch: 0, f_min, f_max
min = Get minimum:  0, 0, "Hertz", "None"
max = Get maximum: 0, 0, "Hertz", "None"
mean = Get mean: 0, 0, "Hertz"
std = Get standard deviation: 0, 0, "Hertz"
writeFileLine: file_path_no_ext$ + ".Pitch", fixed$ (min, 2), "_", fixed$ (max, 2), "_", fixed$ (mean, 2), "_", fixed$ (std, 2)
