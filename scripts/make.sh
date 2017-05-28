cd `dirname $0`
if [[ $mend ]] ; then
    . ~/.e
    mend
fi
if [[ $clean ]] ; then
    rm -rf build-luarocks
    rm -rf /home/graehl/torch/install/lib/luarocks/rocks/fairseq/scm-1/bin/fairseq
    rm -rf /home/graehl/torch/install/lib/luarocks/rocks/fairseq/scm-1/lua/fairseq
    rm -f /home/graehl/torch/install/lib/luarocks/rocks/fairseq/scm-1/lib/libfairseq_clib.so
fi
~/torch/install/bin/luarocks make rocks/fairseq-scm-1.rockspec
