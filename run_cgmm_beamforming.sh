#!/bin/bash
# modified from local/run_beamform_6ch_track.sh
# wujian@17.11.4

. ./path.sh 

nj=10
cmd=run.pl

. utils/parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Wrong #arguments ($#, expected 2)"
   echo "Usage: local/run_cgmm_beamforimg.sh [options] <wav-in-dir> <wav-out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   exit 1;
fi

src_dir=$1
dst_dir=$2
cgmm_dir='data/cgmm'

[ -d $cgmm_dir ] && rm -rf $cgmm_dir && echo "$0: $cgmm_dir existed, regenerated it."
mkdir -p $cgmm_dir

ln -s $(cd $src_dir; pwd) $cgmm_dir/6ch || exit 1
ln -s $(cd $dst_dir; pwd) $cgmm_dir/dst || exit 1

prefix_all=$cgmm_dir/prefix
find $src_dir/{dt,et}*{simu,real}/ | grep CH1.wav \
    | awk -F '/' '{print $(NF-1) "/" $NF}' | sed -e "s/\.CH1\.wav//" | sort > $prefix_all

split_list=""
for n in $(seq $nj); do
    split_list="$split_list $prefix_all.$n"
done

utils/split_scp.pl $prefix_all $split_list || exit 1
func_path=$(cd local; pwd)

for n in $(seq $nj); do
    cat << EOF > $cgmm_dir/run_cgmm_$n.m

addpath('$func_path');

src_dir = '6ch';
dst_dir = 'dst';

prefix_list = textread('prefix.$n', '%s');
prefix_size = size(prefix_list);

for i = 1: prefix_size(1)
    prefix = char(prefix_list(i))
    in  = [src_dir '/' prefix];
    out = [dst_dir '/' prefix];
    apply_cgmm_beamforming(in, out);
end
EOF
done

for x in $(awk -F '/' '{print $1}' $prefix_all | sort | uniq); do
    mkdir -p $dst_dir/$x
done

cd $cgmm_dir

$cmd JOB=1:$nj log/beamform.JOB.log \
   matlab -nojvm -nodisplay -nosplash -r run_cgmm_JOB

echo "$0: run CGMM done"
