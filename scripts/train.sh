#!/bin/bash
d=`dirname $0`
set -e
. srctrglang.sh
. $d/common.sh

main() {
    if [[ -f STOP ]] ; then
        return 1
    fi
    set -e
    if [[ $align ]] ; then
        trainalignfile=`$d/symalign.sh union $TEXT`
    fi
    trained=
    config
    if [[ -d $1 ]] ; then
        resume=1
        trained=$1
    fi
    if [[ $threshold = 1 ]] ; then
        echo "warning: threshold=n means treat counts less than n as threshold. so 1 means no unks"
    fi
    if [[ $redoall || $reprocess || ! -d $bindatadir ]] ; then
        echo preprocess $TEXT to binary $bindatadir
        mkdir -p $bindatadir
        for f in train valid test; do
            for l in $srclang $trglang; do
                ls -l $TEXT/$f.$l
                [[ -f $TEXT/$f.$l ]] || return 1
            done
        done
        fairseq preprocess $langs -trainpref $TEXT/train -validpref $TEXT/valid -testpref $TEXT/test \
                -thresholdsrc $thresholdsrc -thresholdtgt $threshold $alignarg -destdir $bindatadir 2>&1 | tee $bindatadir/preprocess.log
        for f in $bindatadir/*.th7; do
            ls -l $f
            [[ -f $f ]] || return 1
        done
    fi
    mkdir -p $trained
    echo training $TEXT to model $trained
    log=$trained/log
    #typicalcmd="fairseq train -sourcelang jp -targetlang en -datadir data-bin/tokenized.kftt.jp-en-threshold2 -savedir trainings.jp-en.2/fconv-i768-o256.h384.o2x1536.m2x768.11-3.9-3.c3.-1.d.1_29656 -model fconv -nembed 768 -noutembed 256 -fconv_nhids 384,384,384,384,384,384,384,768,768,1536,1536 -fconv_nlmhids 384,384,384,384,384,768,768,1536,1536 -fconv_kwidths 3,3,3,3,3,3,3,3,3,1,1 -fconv_klmwidths 3,3,3,3,3,3,3,1,1 -dropout .1 -nenclayer 11 -nlayer 9 -attnlayers -1 -kwidth 3 -klmwidth 3 -nhid 384 -nagglayer 3 -topnalign 100 -optim nag -timeavg -lr 0.25 -momentum 0.99 -clip 0.1 -bptt 0 -minepochtoanneal 12 -maxepoch 30 -batchsize 64 -maxbatch 1536 -seed 29656 -ndatathreads 4 -log -maxsourcelen 175"
    cmd="fairseq train $langs -datadir $bindatadir -savedir $trained $optimize $commonoptimize -maxsourcelen $maxsourcelen"
    echo $cmd
    traindone=$log.done
    if [[ $redoall || $retrain || ! -f $traindone ]] ; then
        echo $cmd > $log.cmd
        time $cmd 2>&1 | tee $log
        touch $traindone
        ls -l $trained
        tail $log
    fi
    if [[ $resume ]] ; then
         time $cmd 2>&1 | tee $log.resume
    fi
    if [[ $redoall || $reopt || ! -s $optmodel ]] ; then
        fairseq optimize-fconv -input_model $model -output_model $optmodel
        ls -l $optmodel
    fi
    lenpen=15 beam=16 seed=$seed $d/eval.sh $trained
}
main "$@" && exit 0 || exit 1
