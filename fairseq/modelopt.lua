local plpath = require 'pl.path'
local search = require 'fairseq.search'
local tds = require 'tds'

local cmdlineopt = {}

function cmdlineopt.addopt(cmd)
    cmd:option('-sourcelang', 'de', 'source language')
    cmd:option('-targetlang', 'en', 'target language')
    cmd:option('-datadir', 'data-bin')
    cmd:option('-sourcedict', '', 'source language dictionary (empty => datadir/dict.[sourcelang].th7)')
    cmd:option('-targetdict', '', 'target language dictionary (empty => datadir/dict.[targetlang].th7)')
    cmd:option('-minlen', 1, 'minimum length of generated hypotheses')
    cmd:option('-maxlen', 500, 'maximum length of generated hypotheses')
    cmd:option('-topnalign', 100, 'the number of the most common alignments to use')
    cmd:option('-freqthreshold', 0,
        'the minimum frequency for an alignment candidate in order' ..
        'to be considered (default no limit)')
    cmd:option('-aligndictpath', '', 'path to an alignment dictionary (optional)')
    cmd:option('-nmostcommon', 500,
    'the number of most common words to keep when using alignment')

end

local function sourcedict(config)
    return (config.sourcedict and config.sourcedict ~= '') and config.sourcedict or plpath.join(config.datadir, 'dict.' .. config.sourcelang .. '.th7')
end

local function targetdict(config)
    return (config.targetdict and config.targetdict ~= '') and config.targetdict or plpath.join(config.datadir, 'dict.' .. config.targetlang .. '.th7')
end

function cmdlineopt.loadvocab(config)
    srcd = sourcedict(config)
    print(string.format('| [%s] Dictionary file %s', config.sourcelang, srcd))
    config.srcdict = torch.load(srcd)
    print(string.format('| [%s] Dictionary: %d types', config.sourcelang,
        config.srcdict:size()))
    d = targetdict(config)
    print(string.format('| [%s] Dictionary file %s', config.targetlang, d))
    config.dict = torch.load(d)
    print(string.format('| [%s] Dictionary: %d types', config.targetlang,
        config.dict:size()))

    if config.aligndictpath ~= '' then
        config.aligndict = tnt.IndexedDatasetReader{
            indexfilename = config.aligndictpath .. '.idx',
            datafilename = config.aligndictpath .. '.bin',
            mmap = true,
            mmapidx = true,
        }
        config.nmostcommon = math.max(config.nmostcommon, config.dict.nspecial)
        config.nmostcommon = math.min(config.nmostcommon, config.dict:size())
    end

    local vocab = nil
    if config.vocab and config.vocab ~= '' then
        vocab = tds.Hash()
        local fd = io.open(config.vocab)
        while true do
            local line = fd:read()
            if line == nil then
                break
            end
            -- Add word on this line together with all prefixes
            for i = 1, line:len() do
                vocab[line:sub(1, i)] = 1
            end
        end
    end

    return vocab
end

function cmdlineopt.loadmodel(config)
    local model
    if config.model ~= '' then
        model = mutils.loadLegacyModel(config.path, config.model)
    else
        model = require(
            'fairseq.models.ensemble_model'
        ).new(config)
        if config.fconvfast then
            local nfconv = 0
            for _, fconv in ipairs(model.models) do
                if torch.typename(fconv) == 'FConvModel' then
                    fconv:makeDecoderFast()
                    nfconv = nfconv + 1
                end
            end
            assert(nfconv > 0, '-fconvfast requires an fconv model in the ensemble')
        end
    end
    return model
end

function cmdlineopt.searcher(model, config)
    return search.beam{
         ttype = model:type(),
         dict = config.dict,
         srcdict = config.srcdict,
         beam = config.beam,
         lenPenalty = config.lenpen,
         unkPenalty = config.unkpen,
         subwordPenalty = config.subwordpen,
         coveragePenalty = config.covpen,
         vocab = vocab,
         maxlenratio = config.maxlenratio,
         -- subwordContSuffix = config.subwordsuffix,
    }
end

return cmdlineopt
