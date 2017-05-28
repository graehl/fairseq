set +x
. ~/torch/install/bin/torch-activate
if [[ $rebuild ]] ; then
    (cd ~/fairseq;clean=1 ./make.sh)
    rebuild=
fi
agloss() {
    ag --literal '| '"$1" | perl -ne 'print "$1  $_" if m#\| '"$1"'\s+(\S+)#' | sort -n
}
trainloss() {
    agloss trainloss
}
validloss() {
    agloss validloss
}
echo2() {
    echo "$@" 1>&2
}
error() {
    echo2 "ERROR: $*"
    echo "ERROR: $*"
    exit 1
}
detok() {
    $detokenizer
}
abspath() {
    readlink -nfs "$@"
}
multevalabs() {
    local hyp=$1
    shift
    multevalhome=${multevalhome:-~/multeval}
    (
    cd $multevalhome
    ./multeval.sh eval --refs "$@" --hyps-baseline $hyp --meteor.language ${trglang:-en}
    )
}
multeval() {
    local fs=
    for f in "$@"; do
        fs+=" "$(abspath "$f")
    done
    multevalabs $fs
}
bleuscore() {
    ~/bin/bleu.pl -hyp "$@"
}
grepmultbleu() {
    #n=1            BLEU (s_sel/s_opt/p)   METEOR (s_sel/s_opt/p) TER (s_sel/s_opt/p)    Length (s_sel/s_opt/p)
    #baseline       19.7 (0.6/*/-)         23.8 (0.3/*/-)         71.6 (0.7/*/-)         89.3 (1.2/*/-)
    perl -e '$r="[0-9]+[.][0-9+]"; $s="($r)".q{\s+\([^/]+/[^/]+/[^/]+\)}; print STDERR "BLEU\tMETEOR\tTER \tLength\n";while(<>) { print "$1\t$2\t$3\t$4" if m{^\S+\s+$s\s+$s\s+$s\s+$s\s*$} }' "$@"
}
appendrepn() {
    local n=$1
    shift
    local i=1
    while [[ $i -le $n ]] ; do
        repn+="$1,"
        ((i++))
    done
}
endrepn() {
    repn=${repn%,}
}
optmodelname() {
    local f=$1
    if [[ -d $f ]] ; then
        f=$f/$modelname
    fi
    local b=`basename $f`
    echo `dirname $f`/opt$b
}
mkoptmodel() {
    local model=$1
    optmodel=`optmodelname "$model"`
    if [[ $redoall || $reopt || ! -s $optmodel ]] ; then
        fairseq optimize-fconv -input_model "$model" -output_model "$optmodel" 2>$optmodel.log
        if [[ ! -s $optmodel ]] ; then
            echo2 "couldn't optmodel '$model' - not -fconv ?"
            optmodel="$model"
        else
            [[ $quiet ]] || ls -l $optmodel
            [[ $quiet ]] || cat $optmodel.log
        fi
    fi
}
config() {
    set +x
    langs="-sourcelang $srclang -targetlang $trglang"
    alignsuf=
    if [[ $trainalignfile ]] ; then
        alignsuf=.a
        alignarg="-alignfile $trainalignfile"
    fi
    detokenizer=detokenizer.$trglang
    [[ -x $detokenizer ]] || detokenizer=$TEXT/detokenizer.$trglang
    [[ -x $detokenizer ]] || detokenizer=cat
    bindatadir=data-bin/`basename $TEXT`-threshold${threshold:=2}$alignsuf
    trainings=${trainings:-trainings.$srclang-$trglang.$threshold}

    maxsourcelen=175


    batchsz="${batchsize:=32}.${maxbatch:=1536}.${minepochtoanneal:=15}-${maxepoch:=40}"
    fconvtrained=$trainings/fconv-i${nembed:=768}-o${noutembed:=384}.h${nhid:=768}.o${olayers:=0}x${ohid:=2048}.m${mlayers:=0}x${mhid:=1024}.${nenclayer:=9}-${kwidth:=3}.${nlayer:=6}-${klmwidth:=3}.c${nagglayer:=3}.${attnlayers:=-1}.d${dropout:=.2}_${seed:=$(perl -e 'print int(rand(100000))')}

    encplayers=$((nenclayer-olayers-mlayers))
    encwlayers=$((nenclayer-olayers))
    if [[ $encplayers -lt 0 ]] ; then
        error "nenclayer-olayers-mlayers < 0"
    fi
    players=$((nlayer-olayers-mlayers))
    wlayers=$((nlayer-olayers))
    if [[ $players -lt 0 ]] ; then
        error "nlayer-olayers-mlayers < 0"
    fi
    hidsarg=
    if [[ $mlayers -gt 0 || $olayers -gt 0 ]] ; then
        repn=
        appendrepn $encplayers $nhid
        appendrepn $mlayers $mhid
        appendrepn $olayers $ohid
        endrepn
        encnhids=$repn
        repn=
        appendrepn $players $nhid
        appendrepn $mlayers $mhid
        appendrepn $olayers $ohid
        endrepn
        nhids=$repn
        repn=
        appendrepn $encwlayers $kwidth
        appendrepn $olayers 1
        endrepn
        enckwidths=$repn
        repn=
        appendrepn $wlayers $klmwidth
        appendrepn $olayers 1
        endrepn
        kwidths=$repn
        hidsarg="-fconv_nhids $encnhids -fconv_nlmhids $nhids -fconv_kwidths $enckwidths -fconv_klmwidths $kwidths"
        #echo2 hidsarg: $hidsarg
    fi

    ndatathreads=${ndatathreads:=4}
    commonoptimize="-batchsize $batchsize -maxbatch $maxbatch -seed $seed -ndatathreads $ndatathreads -log "
    nagoptimize="-optim nag -timeavg -lr 0.25 -momentum 0.99 -clip 0.1 -bptt 0 -minepochtoanneal $minepochtoanneal -maxepoch $maxepoch"

    fconvoptimize="-model fconv -nembed $nembed -noutembed $noutembed  $hidsarg -dropout $dropout -nenclayer $nenclayer -nlayer $nlayer -attnlayers $attnlayers -kwidth $kwidth -klmwidth $klmwidth -nhid $nhid -nagglayer $nagglayer -topnalign 100 $nagoptimize"

    blstmtrained=$trainings/blstm.${nhid:=512}.${bbatch:=32}
    blstmoptimize="-model blstm -dropout 0.2 -dropout_hid 0 -batchsize $bbatch -optim adam -lr 0.0003125"

    convtrained=$trainings/blstm.${nhid:=512}.${nenclayer}.${convbatch:=32}
    convoptimize="-model conv -dropout 0.2 -dropout_hid 0 -nenclayer $nenclayer -batchsize $convbatch"

    case ${method:=fconv} in
        fconv*)
            fconvfastarg=-fconvfast
            moptimize=$fconvoptimize
            mtrained=$fconvtrained
            ;;
        blstm*)
            fconvfast=
            moptimize=$blstmoptimize
            mtrained=$blstmtrained
            ;;
        conv*)
            fconvfast=
            moptimize=$convoptimize
            mtrained=$convtrained
            ;;
        *)
            echo method=$method unknown
            exit 1
            ;;
    esac
    trained=${trained:-$mtrained}
    trained=${trained%/model_best.th7}
    trained=${trained%/optmodel_best.th7}
    optimize=${optimize:-$moptimize}
    if [[ $modelepoch && $modelepoch != best && ${modelepoch#epoch} = $modelepoch ]] ; then
        modelepoch="epoch$modelepoch"
    fi
    modelname=model_${modelepoch:=best}.th7
    model=$trained/$modelname
    optmodel=`optmodelname "$model"`
    thresholdsrc=${thresholdsrc:-$threshold}
    if [[ $threshold -lt 2 ]] ; then
        threshold=2
        # we want some target unks always??
    fi
    set -e
}
