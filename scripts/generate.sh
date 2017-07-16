. `dirname $0`/common.sh
swsuffixarg="-subwordsuffix __LW_SW__"
snmt=${snmt:-/home/graehl/subword-nmt}
lang=${lang:-${srclang:-en}}
subwordcodesdir=${subwordcodesdir:-$TEXTDIR}
subwordcodes=${subwordcodes:-subword.codes.$lang}
applybpelang() {
    $snmt/apply_bpe.py $swsuffixarg -c subword.codes.$l --vocabulary-threshold 3 $restrictvocab
}
