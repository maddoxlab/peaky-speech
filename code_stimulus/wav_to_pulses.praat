form Test command line calls
	sentence file_path_no_ext file_name
	real f_min 75.0
	real f_max 600.0
endform
Read from file: file_path_no_ext$ + ".wav"
To PointProcess (periodic, peaks): f_min, f_max, "no", "yes"
Save as text file: file_path_no_ext$ + ".PointProcess"
