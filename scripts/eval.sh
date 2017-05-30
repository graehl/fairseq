#!/bin/bash
d=`dirname $0`
set -e
if [[ ! -f srctrglang.sh ]] ; then
    cd ..
fi
. srctrglang.sh
. $d/generate.sh
. $d/common.sh
main() {
    lenpen=${lenpen:-4}
    outdir=ensemble
    config
    set -e
    if [[ $1 ]] ; then
        repn=
        for f in "$@"; do
            [[ -d $f ]] || f="$trainings/$f"
            [[ -d $f ]]
            outdir+=".$(basename $f)"
            mkoptmodel $f/$modelname
            f=$optmodel
            appendrepn 1 "$f"
        done
        endrepn
        model=$repn
    else
        mkoptmodel $$model
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
    if [[ ! $2 ]] ; then
        outdir=$trained
    fi
    [[ -d $bindatadir ]]
    mkdir -p $outdir
    corps=test
    if [[ $evaldev ]] ; then
        corps="devel $corps"
    fi
    for corpus in $corps; do
        name=eval.$corpus.$modelepoch.${beam:=8}.lp${lenpen:=4}.cov${covpen:=0}.sw${sw:=0}
        testf=${testf:-$outdir/${scorename:-$name}}
        [[ $quiet ]] || echo $outdir $corpus:
        if [[ $redoall || $regenerate || ! -s $testf ]] ; then
            cmd="fairseq generate -ndatathreads ${ndatathreads:-4} $langs -datadir $bindatadir -dataset $corpus -path $model -beam $beam -batchsize $batchsize $fconvfast -maxlen $maxsourcelen -lenpen $lenpen -covpen $covpen $swsuffixarg -subwordpen $sw"
            echo2 $cmd
            if ! $cmd > $testf ; then
                rm $testf
                exit 1
            fi
        fi
        [[ $quiet ]] || tail -6 $testf
        $d/score.sh $testf
    done
}
main "$@" && exit 0 || exit 1
