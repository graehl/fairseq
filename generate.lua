-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Batch hypothesis generation script.
--
--]]

require 'nn'
require 'xlua'
require 'fairseq'
local generateopt = require 'fairseq.generateopt'
local tnt = require 'torchnet'
local tds = require 'tds'
local plpath = require 'pl.path'
local hooks = require 'fairseq.torchnet.hooks'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local clib = require 'fairseq.clib'
local mutils = require 'fairseq.models.utils'
local utils = require 'fairseq.utils'
local pretty = require 'fairseq.text.pretty'

local cmd = torch.CmdLine()
generateopt.addopt(cmd)
cmd:option('-nobleu', false, 'don\'t produce final bleu score')
cmd:option('-batchsize', 16, 'batch size')
cmd:option('-dataset', 'test', 'data subset')
cmd:option('-partial', '1/1',
    'decode only part of the dataset, syntax: part_index/num_parts')
cmd:option('-seed', 1111, 'random number seed (for dataset)')
cmd:option('-ndatathreads', 0, 'number of threads for data preparation')

local cuda = utils.loadCuda()

local config = cmd:parse(arg)
torch.manualSeed(config.seed)
if cuda.cutorch then
    cutorch.manualSeed(config.seed)
end

local function accBleu(beam, dict)
    local scorer = clib.bleu(dict:getPadIndex(), dict:getEosIndex())
    local unkIndex = dict:getUnkIndex()
    local refBuf, hypoBuf = torch.IntTensor(), torch.IntTensor()
    return function(sample, hypos)
        if sample then
            local tgtT = sample.target:t()
            local ref = refBuf:resizeAs(tgtT):copy(tgtT)
                    :apply(function(x)
                        return x == unkIndex and -unkIndex or x
                    end)
            for i = 1, sample.bsz do
                local hypoL = hypos[(i - 1) * beam + 1]
                local hypo = hypoBuf:resize(hypoL:size()):copy(hypoL)
                scorer:add(ref[i], hypo)
            end
        end
        return scorer
    end
end

-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------

local vocab, model, searchf = generateopt.loadVocabModelSearcher(config)

local _, test = data.loadCorpus{config = config, testsets = {config.dataset}}
local dataset = test[config.dataset]

local dict, srcdict = config.dict, config.srcdict
local display = pretty.displayResults(dict, srcdict, config.nbest, config.beam)
local computeSampleStats = hooks.computeSampleStats(dict)

-- Ensure that the model is fully unrolled for the maximum source sentence
-- length in the test set. Lazy unrolling might otherwise distort the generation
-- time measurements.
local maxlen = 1
for samples in dataset() do
    for _, sample in ipairs(samples) do
        maxlen = math.max(maxlen, sample.source:size(1))
    end
end
model:extend(maxlen)

-- allow to decode only part of the set k/N means decode part k of N
local partidx, nparts = config.partial:match('(%d+)/(%d+)')
partidx, nparts = tonumber(partidx), tonumber(nparts)

-- let's decode
local addBleu = accBleu(config.beam, dict)
local accTime = generateopt.accTime
local addTime = accTime()
local timer = torch.Timer()
local nsents, ntoks, nbatch = 0, 0, 0
local state = {}
for samples in dataset() do
    if (nbatch % nparts == partidx - 1) then
        assert(#samples == 1, 'can\'t handle multiple samples')
        state.samples = samples
        computeSampleStats(state)
        local sample = state.samples[1]
        local hypos, scores, attns, t = model:generate(config, sample, searchf)
        nsents = nsents + sample.bsz
        ntoks = ntoks + sample.ntokens
        addTime(t)

        -- print results
        if not config.quiet then
            display(sample, hypos, scores, attns)
        end

        -- accumulate bleu
        if (not config.nobleu) then
            addBleu(sample, hypos)
        end
    end
    nbatch = nbatch + 1
end

-- report overall stats
local elapsed = timer:time().real
local statmsg =
    ('| Translated %d sentences (%d tokens) in %.1fs (%.2f tokens/s)')
    :format(nsents, ntoks, elapsed, ntoks / elapsed)
if state.dictstats then
    local avg = state.dictstats.size / state.dictstats.n
    statmsg = ('%s with avg dict of size %.1f'):format(statmsg, avg)
end
print(statmsg)

local timings = '| Timings:'
local totalTime = addTime()
for k, v in pairs(totalTime) do
    local percent = 100 * v.real / elapsed
    timings = ('%s %s %.1fs (%.1f%%),'):format(timings, k, v.real, percent)
end
print(timings:sub(1, -2))

if not config.nobleu then
    local bleu = addBleu()
    print(('| %s'):format(bleu:resultString()))
end
