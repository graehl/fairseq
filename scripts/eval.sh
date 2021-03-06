#!/bin/bash
d=`dirname $0`
set -e
if [[ ! -f srctrglang.sh ]] ; then
    cd ..
fi
. srctrglang.sh
. $d/common.sh
config
. $d/generate.sh
nickname() {
    perl -e '$_=shift;$_ = $1 if /_([0-9]+)$/; print $_' "$1"
}
main() {
    outdir=ensemble
    config
    set -e
    fullmodels=
    models "$@"
    if [[ ! $2 ]] ; then
        outdir=$lastmodeldir
    fi
    if [[ $testf ]] ; then
        outdir=`dirname $testf`
    fi
    [[ -d $bindatadir ]]
    mkdir -p $outdir
    echo "$fullmodels" > $outdir/fullmodels.ls
    corps=test
    if [[ $evaldevonly ]] ; then
        corps="valid"
    fi
    if [[ $evaldev ]] ; then
        corps="valid $corps"
    fi
    for corpus in $corps; do
        name=eval.$corpus.$modelepoch.${beam:=16}.lp${lenpen:=15}.cov${covpen:=0}.sw${sw:=0}
        testf=${testf:-$outdir/${scorename:-$name}}
        [[ $quiet ]] || echo $outdir $corpus:
        if [[ $redoall || $regenerate || ! -s $testf ]] ; then
            if [[ ! $nogenerate ]] ; then
                genargs="$langs -path $model -beam $beam $fconvfast -maxlen $maxsourcelen -lenpen $lenpen -covpen $covpen $swsuffixarg -subwordpen $sw -datadir $bindatadir"
                generating=$generatepre
                if [[ ! -s $TESTSRCTOKENS ]] ; then
                    echo2 no TESTSRCTOKENS=$TESTSRCTOKENS
                    generating=1
                fi
                if [[ $generating || $corpus = valid ]] ; then
                    cmd="fairseq generate $genargs -batchsize $batchsize -ndatathreads ${ndatathreads:-4} -dataset $corpus"
                else
                    cmd="fairseq generate-lines -pretty $genargs -input $TESTSRCTOKENS"
                fi
                echo "$cmd" > $testf.log
                echo2 "$cmd"
                if ! time $cmd > $testf ; then
                    rm $testf
                    exit 1
                fi
                egrep '^\|' $testf > $testf.perf
            fi
        fi
        [[ $quiet ]] || tail $testf
        if [[ $corpus = test ]] ; then
            detoklc=$TESTDETOKLC
            detokmixed=$TESTDETOK
            set -e
            [[ -s $detoklc ]]
            [[ -s $detokmixed ]]
        else
            detoklc=$VALIDDETOKLC
            detokmixed=$VALIDDETOK
        fi
        detoklc=$detoklc detokmixed=$detokmixed seed=$seed $d/score.sh $testf
    done
}
main "$@" && exit 0 || exit 1
