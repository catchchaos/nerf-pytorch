if [ $# -ne 4 ]
	then
	echo 'Invalid #args, please pass 4 nerf output directories to make a collage'
	exit 1
fi

COLOR=blue

ffmpeg -i $1/%04d.png -i $2/%04d.png -i $2/%04d.png -i $3/%04d.png -filter_complex \
"[0]drawtext=text=$1:fontsize=20:x=10:y=10:fontcolor=$COLOR[v0];
 [1]drawtext=text=$2:fontsize=20:x=10:y=10:fontcolor=$COLOR[v1];
 [2]drawtext=text=$3:fontsize=20:x=10:y=10:fontcolor=$COLOR[v2];
 [3]drawtext=text=$4:fontsize=20:x=10:y=10:fontcolor=$COLOR[v3];
 [v0][v1][v2][v3]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map ["v"] \
-c:v libx264 -preset veryslow -crf 18 combined.avi
