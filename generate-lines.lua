-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Hypothesis generation script with text file input, processed line-by-line.
-- By default, this will run in interactive mode.
--
--]]

require 'fairseq'
local generateopt = require 'fairseq.generateopt'

local tnt = require 'torchnet'
local tds = require 'tds'
local argcheck = require 'argcheck'
local plstringx = require 'pl.stringx'
local data = require 'fairseq.torchnet.data'
local search = require 'fairseq.search'
local tokenizer = require 'fairseq.text.tokenizer'
local mutils = require 'fairseq.models.utils'
local pretty = require 'fairseq.text.pretty'

local cmd = torch.CmdLine()
generateopt.addopt(cmd)
cmd:option('-input', '-', 'source language input text file')
cmd:option('-visdom', '', 'visualize with visdom: (host:port)')
cmd:option('-pretty', false, 'write output in fairseq generate "pretty" format i.e. with -[input#] e.g. H-1')

local config = cmd:parse(arg)

-------------------------------------------------------------------
-- Load data
-------------------------------------------------------------------
local vocab, model, searchf = generateopt.loadVocabModelSearcher(config)

local TextFileIterator, _ =
    torch.class('tnt.TextFileIterator', 'tnt.DatasetIterator', tnt)

TextFileIterator.__init = argcheck{
    {name='self', type='tnt.TextFileIterator'},
    {name='path', type='string'},
    {name='transform', type='function',
        default=function(sample) return sample end},
    call = function(self, path, transform)
        function self.run()
            local fd
            if path == '-' then
                fd = io.stdin
            else
                fd = io.open(path)
            end
            return function()
                if torch.isatty(fd) then
                    io.stdout:write('> ')
                    io.stdout:flush()
                end
                local line = fd:read()
                if line ~= nil then
                    return transform(line)
                elseif fd ~= io.stdin then
                    fd:close()
                end
            end
        end
    end
}

local dataset = tnt.DatasetIterator{
    iterator = tnt.TextFileIterator{
        path = config.input,
        transform = function(line)
            return {
                bin = tokenizer.tensorizeString(line, config.srcdict),
                text = line,
            }
        end
    },
    transform = function(sample)
        local source = sample.bin:view(-1, 1):int()
        local sourcePos = data.makePositions(source,
            config.srcdict:getPadIndex()):view(-1, 1)
        local sample = {
            source = source,
            sourcePos = sourcePos,
            text = sample.text,
            target = torch.IntTensor(1, 1), -- a stub
        }
        sample.target[1] = 0 -- stub for pretty.displayResults
        if config.aligndict then
            sample.targetVocab, sample.targetVocabMap,
                sample.targetVocabStats
                    = data.getTargetVocabFromAlignment{
                        dictsize = config.dict:size(),
                        unk = config.dict:getUnkIndex(),
                        aligndict = config.aligndict,
                        set = 'test',
                        source = sample.source,
                        target = sample.target,
                        nmostcommon = config.nmostcommon,
                        topnalign = config.topnalign,
                        freqthreshold = config.freqthreshold,
                    }
        end
        return sample
    end,
}

if config.visdom ~= '' then
    local host, port = table.unpack(plstringx.split(config.visdom, ':'))
    searchf = search.visualize{
        sf = searchf,
        dict = config.dict,
        sourceDict = config.srcdict,
        host = host,
        port = tonumber(port),
    }
end

local dict, srcdict = config.dict, config.srcdict
local display = pretty.displayResults(dict, srcdict, config.nbest, config.beam)
local eos = dict:getSymbol(dict:getEosIndex())
local seos = srcdict:getSymbol(srcdict:getEosIndex())
local unk = dict:getSymbol(dict:getUnkIndex())

-- Select unknown token for reference that can't be produced by the model so
-- that the program output can be scored correctly.
local runk = unk
repeat
    runk = string.format('<%s>', runk)
until dict:getIndex(runk) == dict:getUnkIndex()

local accTime = generateopt.accTime

local addTime = accTime()
local timer = torch.Timer()
local nsents, ntoks, nbatch = 0, 0, 0

for sample in dataset() do
    nsents = nsents + 1
    sample.bsz = 1
    local hypos, scores, attns, t = model:generate(config, sample, searchf)
    addTime(t)

    -- Print results
    local sourceString = config.srcdict:getString(sample.source:t()[1])
    sourceString = sourceString:gsub(seos .. '.*', '')
    for i in string.gmatch(sourceString, "%S+") do
       ntoks = ntoks + 1
    end

    if not config.quiet then
       if config.pretty then
          sample.index = { nsents }
          display(sample, hypos, scores, attns)
       else
           print('S', sourceString)
           print('O', sample.text)

           for i = 1, math.min(config.nbest, config.beam) do
               local hypo = config.dict:getString(hypos[i]):gsub(eos .. '.*', '')
               print('H', scores[i], hypo)
               -- NOTE: This will print #hypo + 1 attention maxima. The last one is the
               -- attention that was used to generate the <eos> symbol.
               local _, maxattns = torch.max(attns[i], 2)
               print('A', table.concat(maxattns:squeeze(2):totable(), ' '))
           end
       end
    end


    io.stdout:flush()
    collectgarbage()
end

-- report overall stats
local elapsed = timer:time().real
local statmsg =
    ('| Translated %d sentences (%d tokens) in %.1fs (%.2f tokens/s)')
    :format(nsents, ntoks, elapsed, ntoks / elapsed)
print(statmsg)
local timings = '| Timings:'
local totalTime = addTime()
for k, v in pairs(totalTime) do
    local percent = 100 * v.real / elapsed
    timings = ('%s %s %.1fs (%.1f%%),'):format(timings, k, v.real, percent)
end
print(timings:sub(1, -2))

-- TODO: parallel generate-lines (if nthreads > 1 - batched / noninteractive / with timeout?)
        -- local makePipeline = function()
        --     -- Note that we create init, samplesize and merge functions here,
        --     -- becase we want to provide thread locality for their upvalues.
        --     local params = {
        --         dataset = init(set)(),
        --         samplesize = samplesize(set),
        --         merge = merge(set),
        --         batchsize = config.batchsize,
        --     }
        --     if test then
        --         return makeTestDataPipeline(params)
        --     else
        --         local ds = makeTrainingDataPipeline(params)
        --         -- Attach a function to set the random seed. This dataset will
        --         -- live in a seprate thread, and this is a convenient way to
        --         -- initialize the RNG local to that thread.
        --         ds.setRandomSeed = function(self, seed)
        --             torch.manualSeed(seed)
        --         end
        --         return ds
        --     end
        -- end

        -- local makeIterator = function()
        --     local it
        --     if config.nthread == 0 then
        --         it = tnt.DatasetIterator{
        --             dataset = makePipeline(),
        --         }
        --     else
        --         it = tnt.ParallelDatasetIterator{
        --             nthread = config.nthread,
        --             init = function()
        --                 require 'torchnet'
        --                 tds = require 'tds'
        --                 require 'fairseq'
        --                 if seed then
        --                     torch.manualSeed(seed)
        --                 end
        --             end,
        --             closure = makePipeline,
        --             ordered = true,
        --         }
        --     end
