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
    (
        xmttrgnorm
        #/home/graehl/bin/Utf8Normalize.sh --nfkd
    )
    #2>/dev/null
}
cutlastof3() {
    cut -f2- "$@" |
    cut -f2- "$@"
}
psidebyside() {
    perl -e '$pkeep=shift;
@f=map {my $fh; open $fh,"<",$_ or die "open $_"; $fh} @ARGV;
$n=0;
for(;;) {
    ++$n; @l = map { scalar <$_> } @f; last if !defined($l[0]);
    next unless rand() < $pkeep; print "$n\n"; print $_ for (@l);
}
' "$@"
}
sidebyside() {
    psidebyside 1 "$@"
}
main() {
    set -e
    G=$1
    G=${G%.score}
    [[ -f $G ]]
    testscore=$G.score
    . `dirname $0`/common.sh
    if [[ $redoall || ! $norescore || ! -s $testscore ]] ; then
        dconf=.
        [[ -f $dconf/config.sh ]] || dconf+=..
        if [[ -f $dconf/config.sh ]] ; then
            . $dconf/config.sh
            ls -l ${TESTTOKENS:?$dconf/srctrglang.sh no TESTTOKENS set}
        fi
        if [[ $redoall || $resplit || $retokbleu || ! -s $G.sys.bpe || ! -s $G.ref.bpe ]] ; then
#sort is because:
#S-197   to the participating members
#T-197   الى السادة المشتركين
#H-197   -0.8388689160347        للدول المشاركة
#A-197   1 1 4
            grep ^H $G | cut -c3- | sort -n > $G.H
            grep ^T $G | cut -c3- | sort -n > $G.T
            cutlastof3 < $G.H | normalize > $G.sys.bpe
            if [[ -s $G.T ]] ; then
                cutlastof3 < $G.T | normalize > $G.ref.bpe
            elif [[ -s $TESTTOKENS ]] ; then
                rm $G.T
                normalize < $TESTTOKENS > $G.ref.bpe
            fi
            [[ $quiet ]] || wc $TESTTOKENS $G.H $G.sys.bpe $G.ref.bpe
        fi
        for f in sys ref; do
            unbpe < $G.$f.bpe > $G.$f
            [[ $quiet ]] || wc $G.$f $G.$f.bpe
        done
#        savecmd $testscore fairseq score -ignore_case -sys $G.sys -ref $G.ref
#        savecmd $testscore.bpe fairseq score -ignore_case -sys $G.sys.bpe -ref $G.ref.bpe
        savecmd $testscore bleuscore $G.sys.bpe $G.ref.bpe
        savecmd $testscore.bpe bleuscore $G.sys $G.ref
        [[ $quiet ]] || echo `pwd`/$testscore
        [[ $quiet ]] || cat $testscore{,.bpe}
        if [[ -f $dconf/config.sh ]] ; then
            echo toklc=${toklc:=${TESTTOKLC:-test.trg.tok.lc}}
            echo detoklc=${detoklc:=${TESTDETOKLC:-test.trg.lc}}
            echo detokmixed=${detokmixed:=${TESTDETOK:-test.trg}}
            detoklcin=$detoklc
            [[ -s $detoklcin ]] || lcutf8 < $detokmixed > $detoklc
            detokmixedin=$detokmixed
            detoklc=$detoklc.norm
            detokmixed=$detokmixed.norm
            toklcin=$toklc
            toklc=$toklc.norm
            lcutf8 < $G.sys > $G.sys.lc
            if [[ $resplit || ! -s $detoklc || ! -s $detokmixed || ! -s $toklc ]] ; then
                normalize < $toklcin > $toklc
                normalize < $detoklcin > $detoklc
                normalize < $detokmixedin > $detokmixed
            fi
            scoremixed=$testscore.detok.mixedBLEU
            score=$testscore.detok.lcBLEU
            refscore=$testscore.detok.ref.lcBLEU
            tokscore=$testscore.tok.lcBLEU
            if [[ $redoall || $retokbleu || ! -s $score || ! -s $scoremixed ]] ; then
                 for f in sys ref; do
                     if [[ $retokbleu || $f = sys || ! -s $G.$f.detok.lc ]] ; then
                         #xmtiopost $G.$f.bpe $G.$f.detok >/dev/null 2>/dev/null
                         detoklwat $G.$f | normalize > $G.$f.detok
                         #xmtiolc $G.$f.detok $G.$f.detok.lc >/dev/null 2>/dev/null
                         lcutf8 $G.$f.detok | normalize > $G.$f.detok.lc
                     fi
                 done
                 [[ -s $toklc ]] || toklc=$G.ref.lc
                 [[ -s $detoklc ]] || detoklc=$G.ref.detok.lc
                 [[ -s $detokmixed ]] || detokmixed=$G.ref.detok
                 savecmd $tokscore bleuscore $G.sys.lc $toklc
                 savecmd $refscore bleuscore $G.sys.detok.lc $detoklc
                 #2>/dev/null
                 lcs="$G.sys.detok.lc $G.ref.detok.lc"
                 sidebyside $TESTSRC $lcs > $G.lc.sidebyside
                 savecmd $score bleuscore $lcs 2>/dev/null
                 #if [[ -f $detokmixed ]] ; then
                 mixeds="$G.sys.detok $G.ref.detok"
                 savecmd $scoremixed bleuscore $mixeds
                 sidebyside $TESTSRC $mixeds > $G.sidebyside
                 #$detokmixed
                 #2>/dev/null
                 #[[ $quiet ]] ||         echo `pwd`/$score
                 #[[ $quiet ]] ||         cat $score
            fi
        fi
        multscore=$testscore.multeval
        multscoredetok=$testscore.multeval.detok
        multscoredetoklc=$testscore.multeval.detok.lc
        if false && [[ $redoall || $rescore || ! -s $multscore ]] ; then
            savecmd $multscore multeval1ref $G.ref $G.sys
            #savecmd $multscoredetok multeval1ref $detokmixed $G.sys.detok
            #savecmd $multscoredetoklc multeval1ref $detoklc $G.sys.detok.lc
        fi
        #[[ $quiet ]] ||             echo `pwd`/$multscore `pwd`/$score
        #[[ $quiet ]] ||             tail $multscore* $score*
        set -x
        tail -n 4 $G*.sidebyside $G.*BLEU $G.perf
        grep BLEU `dirname $G`/*.mixedBLEU | sort -k 3 -n
        grep BLEU $scoremixed
        #$multscoredetok* $multscoredetoklc*
    fi
}
main "$@" && exit 0 || exit 1
