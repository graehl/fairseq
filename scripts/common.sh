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
bleuscore() {
    ~/bin/bleu.pl -hyp "$@"
}
grepmultbleu() {
    #n=1            BLEU (s_sel/s_opt/p)   METEOR (s_sel/s_opt/p) TER (s_sel/s_opt/p)    Length (s_sel/s_opt/p)
    #baseline       19.7 (0.6/*/-)         23.8 (0.3/*/-)         71.6 (0.7/*/-)         89.3 (1.2/*/-)
    perl -e '$r="[0-9]+[.][0-9+]"; $s="($r)".q{\s+\([^/]+/[^/]+/[^/]+\)}; while(<>) { print "$1\t$2\t$3\t$4" if m{^\S+\s+$s\s+$s\s+$s\s+$s\s*$} }' "$@"
    #print STDERR "BLEU\tMETEOR\tTER \tLength\n"
}
grepbleu() {
    #eval.test.best.12.lp8.cov0.sw0.sys.detok.lc	 BLEUr1n4[%] 21.4 brevityPenalty: 0.9257 lengthRatio: 0.9283  95%-conf.: 20.3 - 22.51 delta: 1.1
    perl -ne 'BEGIN{print STDERR "BLEU\tLR\n"}; print "$1\t",sprintf("%.2f",$2) if /\QBLEUr1n4[%]\E ([0-9.]+) .* lengthRatio: ([0-9.]+) /' "$@"
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
    if [[ ${threshold:=2} -lt 2 ]] ; then
        threshold=2
        # we want some target unks always. if you use 1 then target vocab has 3 items in it (oops)
    fi
    if [[ ${thresholdsrc:=${threshold}} -lt 2 ]] ; then
        thresholdsrc=2
    fi
    thresholdname=$threshold
    if [[ $thresholdsrc != $threshold ]] ; then
        thresholdname+="-s$thresholdsrc"
    fi
    bindatadir=data-bin/`basename $TEXT`-threshold$thresholdname$alignsuf
    trainings=${trainings:-trainings.$srclang-$trglang.$thresholdname}

    maxsourcelen=${maxsourcelen:-160}

    batchsz="${batchsize:=64}.${maxbatch:=1024}.${minepochtoanneal:=15}-${maxepoch:=40}.p${patience:=2}"

    #increasing nembed nhid to 640 or 768 also works well. noutembed not so much? more training data supports larger network training w/o overfitting

    if [[ $olayers || $mlayers ]] ; then
        omlayers=".o${olayers:=0}x${ohid:=2048}.m${mlayers:=0}x${mhid:=1024}"
    else
        olayers=0
        mlayers=0
    fi

    fconvtrained=$trainings/fconv-i${nembed:=512}-o${noutembed:=384}.h${nhid:=512}$omlayers.${nenclayer:=9}-${kwidth:=3}.${nlayer:=6}-${klmwidth:=3}.c${nagglayer:=3}.${attnlayers:=-1}.d${dropout:=.2}_${seed:=$(perl -e 'print int(rand(100000))')}

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

    fconvoptimize="-model fconv -nembed $nembed -noutembed $noutembed  $hidsarg -dropout $dropout -nenclayer $nenclayer -nlayer $nlayer -attnlayers $attnlayers -kwidth $kwidth -klmwidth $klmwidth -nhid $nhid -nagglayer $nagglayer -topnalign 100 $nagoptimize -patience $patience"

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
    set -e
}
modelbest=model_best.th7
optmodelbest=opt$modelbest
findunfinished() {
    for f in ${1:-trainings.jp-en.2}/*; do
        if [[ -d $f && ! -s $f/$modelbest && ! -s $f/$optmodelbest ]] ; then
            echo $f
        fi
    done
}
renamerm() {
    perl -e '$_=shift;$rm=shift;$was=$_;rename $was,$_ if (s/\Q$rm\E//)' "$@"
}
rmo0m0() {
    for f in "$@"; do
        renamerm "$f" ".o0x2048.m0x1024"
    done
}


multeval12() {
    (
        [[ -s srctrglang.sh ]] && . srctrglang.sh
        ref=${3:-$TESTDETOKLC}
        set -e
        set -x
        [[ -s $ref ]]
        a=$1
        [[ $a ]]
        b=$2
        [[ $b ]]
        suf=${4:-sys.detok.lc}
        as=$a*/*$suf
        bs=$b*/*$suf
        allnonempty $as
        allnonempty $bs
        multeval --refs "$ref" --hyps-baseline $as --hyps-sys1 $bs 2>&1 | tee multeval.`basename $a`.`basename $b`.log
    )
}
multeval1ref() {
    local ref=$1
    shift
    multeval --refs "$ref" --hyps-baseline "$@"
}
multevalabs() {
    multevalhome=${multevalhome:-~/multeval}
    (
        cd $multevalhome
        ./multeval.sh eval "$@" --meteor.language ${trglang:-en}
    )
}
multeval() {
    local fs=
    for f in "$@"; do
        if [[ -s "$f" && `dirname $f` != . ]] ; then
            f=$(abspath "$f")
        fi
        fs+=" $f"
    done
    multevalabs $fs
}
models() {
    if [[ $1 ]] ; then
        repn=
        for f in "$@"; do
            [[ -d $f ]] || f="$trainings/$f"
            [[ -f $f/$modelbest ]] || f="$trainings/$f"
            [[ -d $f ]]
            [[ -f $f/$modelbest ]]
            basef=`basename $f`
            lastmodeldir=$f
            outdir+=".$(nickname $basef)"
            fullmodels+=" $basef"
            mkoptmodel $f/$modelname
            f=$optmodel
            appendrepn 1 "$f"
        done
        endrepn
        model=$repn
    else
        fullmodels=$model
        mkoptmodel $model
        if [[ -s $optmodel ]] ; then
            model=$optmodel
            fconvfast=
        elif [[ ${nofconvfast:=1} = 1 ]] ; then
            fconvfast=
        else
            fconvfast=$fconvfastarg
        fi
        [[ -f $model ]]
    fi
    echo ${beam:=16}.lp${lenpen:=15}.cov${covpen:=0}.sw${sw:=0} > /dev/null
}
