using Pkg
Pkg.activate(".")

using Revise
using DataFrames, CSV
using ForwardDiff
using NLsolve
using LinearAlgebra
using Optim
using LineSearches
using JuMP
using Ipopt
using Statistics: cov
using Infiltrator
include("src/data.jl")
include("src/estimation.jl")
include("src/estimation.jl")
include("src/share.jl")
include("src/simulate.jl")


df = CSV.read("data/blp_1999_data.csv", DataFrame)

exog = [:const, :hpwt, :air, :mpd, :space, :mpg, :trend]

function makeinstruments(df, exog)
    for z ∈ exog
        own = groupby(df, [:year, :firm_id])
        own = combine(own, z => sum => Symbol(z, "_own"))
        all = groupby(df, [:year])
        all = combine(all, z => sum => Symbol(z, "_other"))
        df = leftjoin(df, own, on=[:year, :firm_id])
        df = leftjoin(df, all, on=[:year])
        df[!, Symbol(z, "_other")] .-= df[!, Symbol(z, "_own")]
        df[!, Symbol(z, "_own")] .-= df[!, z]
    end
    return (df)
end

df = makeinstruments(df, exog);
df[!, :loghpwt] = log.(df[!, :hpwt])
df[!, :logmpg] = log.(df[!, :mpg])
df[!, :logspace] = log.(df[!, :space])
S = 20
xvars = [:price, :const, :hpwt, :air, :mpd, :space]
costvars = [:const, :loghpwt, :air, :logmpg, :logspace];

dat = blpdata(df, :year, :firm_id, :share, xvars, costvars,
    [:const,], [:const],
    randn(length(xvars), S, length(unique(df[!, :year]))))

dat = makeivblp(dat, includeexp=false)

out = estimateRCIVlogit(dat, method=:MPEC, verbose=true, W=I)

v = varianceRCIVlogit(out.β, out.σ, dat, W=I)

@show [out.β out.σ];
@show sqrt.(diag(v.Σ));