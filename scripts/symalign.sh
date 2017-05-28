. ~/torch/install/bin/torch-activate
if [[ -f srctrglang.sh ]] ; then
    . srctrglang.sh
fi
moses=~/mosesdecoder
fastalign=~/fast_align
fairseq=~/fairseq
sym=${1:-union}
odir=${2:-$TEXT}
src=TEXT/train.$srclangtrain
trg=$TEXT/train.$trglang
mkdir -p $odir
$fairseq/scripts/build_sym_alignment.py --fast_align_dir=$fastalign --mosesdecoder_dir=$moses --sym_heuristic=$sym --source_file=$src --target_file=$trg --output_dir=$odir 1>&2
aligned=$odir/aligned.$sym
th $fairseq/scripts/makealigndict.lua -source $src -target $trg -alignment $aligned 1>&2
