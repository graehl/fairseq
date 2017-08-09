local modelopt = require 'fairseq.modelopt'
local generateopt = {}

function generateopt.addopt(cmd)
    modelopt.addopt(cmd)
    cmd:option('-path', 'model1.th7,model2.th7', 'path to saved model(s)')
    cmd:option('-beam', 1, 'search beam width')
    cmd:option('-nbest', 1, 'number of candidate hypotheses')
    cmd:option('-maxlenratio', 0, 'if >0, maximum number of generated tokens per input token')
    cmd:option('-fconvfast', false, 'make fconv model faster')
    cmd:option('-model', '', 'model type for legacy models')
    cmd:option('-vocab', '', 'restrict output to target vocab')

    --- following affect the model but aren't used in train.lua (TODO - should
    --- they be used for early stopping on dev?)
    cmd:option('-lenpen', 1,
        'length penalty: <1.0 favors shorter, >1.0 favors longer sentences')
    cmd:option('-unkpen', 0,
        'unknown word penalty: <0 produces more, >0 produces less unknown words')
    cmd:option('-subwordpen', 0,
        'subword penalty: <0 favors longer, >0 favors shorter words')
    cmd:option('-subwordsuffix', '__LW_SW__',
        'for subwordpen: subword elements end in this suffix')
    cmd:option('-covpen', 0,
               'coverage penalty: favor hypotheses that cover all source tokens')

    cmd:option('-quiet', false, 'don\'t print generated text')
end

function generateopt.loadVocabModelSearcher(config)
    -- return (vocab, model, searcher)
    local vocab = modelopt.loadvocab(config)
    local model = modelopt.loadmodel(config)
    local searcher = modelopt.searcher(model, config)
    return vocab, model, searcher
end

function generateopt.accTime()
    local total = {}
    return function(times)
        for k, v in pairs(times or {}) do
            if not total[k] then
                total[k] = {real = 0, sys = 0, user = 0}
            end
            for l, w in pairs(v) do
                total[k][l] = total[k][l] + w
            end
        end
        return total
    end
end

return generateopt
