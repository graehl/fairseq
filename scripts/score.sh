#!/bin/bash
#outputs: $G.score (fairseq) $G.multeval $G.score.detok.lcBLEU (xmt)
unbpe() {
    sed 's/\Q@@\E //g;s/__LW_SW__ //g;'
}
cleanuplwat() {
    perl -pe 's/(^| )__LW_AT__|\Q_-#-_\E( |$)/$1/g' "$@"
}
detoklwat() {
    perl -pe 's/ ?(?:__LW_AT__|\Q_-#-_\E) ?//go' "$@"
}
perlutf8() {
    local enutf8=${enutf8:-en_US.UTF-8}
    LC_ALL=$enutf8 perl "$@"
}
lcutf8() {
    perlutf8 -pe '$_=lc($_)' "$@"
}
savecmd() {
    local out=$1
    shift
    echo "$*" > $out.cmd
    $* > $out
    cat $out
}
normalize() {
    /home/graehl/bin/Utf8Normalize.sh --nfkd "$@"
    #2>/dev/null
}
main() {
    set -e
    G=$1
    G=${G%.score}
    [[ -f $G ]]
    testscore=$G.score
    if [[ $redoall || ! $norescore || ! -s $testscore ]] ; then
        if [[ $redoall || $resplit || $retokbleu || ! -s $G.sys.bpe || ! -s $G.ref.bpe ]] ; then
            grep ^H $G | cut -c3- | sort -n > $G.H
            grep ^T $G | cut -c3- | sort -n > $G.T
            [[ $quiet ]] || wc -l $G.H $G.T
            cut -f3- < $G.H | normalize > $G.sys.bpe
            cut -f2- < $G.T | normalize > $G.ref.bpe
        fi
        for f in sys ref; do
            unbpe < $G.$f.bpe > $G.$f
            [[ $quiet ]] || wc $G.$f $G.$f.bpe
        done
        fairseq score -sys $G.sys -ref $G.ref > $testscore
        [[ $quiet ]] || echo `pwd`/$testscore
        [[ $quiet ]] || cat $testscore
        . `dirname $0`/common.sh
        dconf=.
        [[ -f $dconf/srctrglang.sh ]] || dconf=..
        if [[ -f $dconf/srctrglang.sh ]] ; then
            . $dconf/srctrglang.sh
            if [[ -f $dconf/xmtconfig.sh ]] ; then
                . $dconf/xmtconfig.sh
                xmtconfig
            fi
            echo detoklc=${detoklc:=${TESTDETOKLC:-test.trg.lc}}
            echo detokmixed=${detokmixed:=${TESTDETOK:-test.trg}}
            [[ -s $detoklcin ]] || lcutf8 < $detokmixed > $detoklc
            detoklcin=$detoklc
            detokmixedin=$detokmixed
            detoklc=$detoklc.norm
            detokmixed=$detokmixed.norm
            if [[ $resplit || ! -s $detoklc || ! -s $detokmixed ]] ; then
                normalize < $detoklcin > $detoklc
                normalize < $detokmixedin > $detokmixed
            fi
            scoremixed=$testscore.detok.mixedBLEU
            score=$testscore.detok.lcBLEU
            refscore=$testscore.detok.ref.lcBLEU
            if [[ $redoall || $retokbleu || ! -s $score || ! -s $scoremixed ]] ; then
                 for f in sys ref; do
                     if [[ $retokbleu || $f = sys || ! -s $G.$f.detok.lc ]] ; then
                         #xmtiopost $G.$f.bpe $G.$f.detok >/dev/null 2>/dev/null
                         detoklwat $G.$f | normalize > $G.$f.detok
                         #xmtiolc $G.$f.detok $G.$f.detok.lc >/dev/null 2>/dev/null
                         lcutf8 $G.$f.detok | normalize > $G.$f.detok.lc
                     fi
                 done
                 [[ -s $detoklc ]] || detoklc=$G.ref.detok.lc
                 [[ -s $detokmixed ]] || detokmixed=$G.ref.detok
                 savecmd $refscore bleuscore $G.sys.detok.lc $detoklc
                     #2>/dev/null
                 savecmd $score bleuscore $G.sys.detok.lc $G.ref.detok.lc 2>/dev/null
                 #if [[ -f $detokmixed ]] ; then
                 savecmd $scoremixed bleuscore $G.sys.detok $detokmixed
                 #2>/dev/null
                 #[[ $quiet ]] ||         echo `pwd`/$score
                 #[[ $quiet ]] ||         cat $score
            fi
        fi
        multscore=$testscore.multeval
        multscoredetok=$testscore.multeval.detok
        multscoredetoklc=$testscore.multeval.detok.lc
        if [[ $redoall || $rescore || ! -s $multscore ]] ; then
            savecmd $multscore multeval1ref $G.ref $G.sys
            #savecmd $multscoredetok multeval1ref $detokmixed $G.sys.detok
            #savecmd $multscoredetoklc multeval1ref $detoklc $G.sys.detok.lc
        fi
        #[[ $quiet ]] ||             echo `pwd`/$multscore `pwd`/$score
        [[ $quiet ]] ||             tail $multscore* $score* $multscoredetok* $multscoredetoklc*
    fi
}
main "$@" && exit 0 || exit 1
