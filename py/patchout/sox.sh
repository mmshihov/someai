search_dir=/home/mikhail/py/work/test-data/audio-wav-16
save_dir=/home/mikhail/py/work/test-data/audio-wav

for entry in "$search_dir"/*
do
	echo "Processing $entry"
	file=$(basename $entry)
	sox $entry -r 20000 -c 1 -b 16 --endian little -e signed-integer $save_dir/$file.wav # trim 0 01:00
done
