-- Copyright (c) 2017-present, Facebook, Inc.
-- All rights reserved.
--
-- This source code is licensed under the license found in the LICENSE file in
-- the root directory of this source tree. An additional grant of patent rights
-- can be found in the PATENTS file in the same directory.
--
--[[
--
-- Command-line script for BLEU scoring.
--
--]]

local tok = require 'fairseq.text.tokenizer'
local plpath = require 'pl.path'
local bleu = require 'fairseq.text.bleu'

local cmd = torch.CmdLine()
cmd:option('-sys', '-', 'system output')
cmd:option('-ref', '', 'references')
cmd:option('-order', 4, 'consider ngrams up to this order')
cmd:option('-ignore_case', false, 'case-insensitive scoring')

local config = cmd:parse(arg)

assert(config.sys == '-' or plpath.exists(config.sys))
local fdsys = config.sys == '-' and io.stdin or io.open(config.sys)
assert(plpath.exists(config.ref))
local fdref = io.open(config.ref)

local function readLine(fd)
    local s = fd:read()
    if s == nil then
        return nil
    end

    if config.ignore_case then
        s = string.lower(s)
    end
    return tok.tokenize(s)
end

local scorer = bleu.scorer(config.order)

local function extralines(fd, name)
    count = 1
    while true do
       local line = readLine(fd)
       if line == nil then
           break
       end
       count = count + 1
    end
    error (tostring(count) .. ' extra lines in ' .. name)
end

-- Process system output and reference file
while true do
    local sysTok = readLine(fdsys)
    local refTok = readLine(fdref)
    if sysTok == nil and refTok ~= nil then
        extralines(fdref, 'reference')
    elseif refTok == nil and sysTok ~= nil then
        extralines(fdsys, 'system')
    elseif sysTok == nil and refTok == nil then
        break
    end

    scorer.update(sysTok, refTok)
end

print(scorer.resultString())
