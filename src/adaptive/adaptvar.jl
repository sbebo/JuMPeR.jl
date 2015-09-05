#-----------------------------------------------------------------------
# JuMPeR  --  JuMP Extension for Robust Optimization
# http://github.com/IainNZ/JuMPeR.jl
#-----------------------------------------------------------------------
# Copyright (c) 2015: Iain Dunning
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#-----------------------------------------------------------------------
# src/adaptive.jl
# Adaptive robust optimization support
#-----------------------------------------------------------------------

export @defAdaptVar

#-----------------------------------------------------------------------
# AdaptiveVariable
type AdaptiveVariable <: JuMP.AbstractJuMPScalar
    m::Model
    id::Int
end
function AdaptiveVariable(m::Model, name::String, lower::Real, upper::Real,
                    cat::Symbol, policy::Symbol,
                    stage::Int, depends_on::Vector{Uncertain})
    rd = getRobust(m)
    rd.numAdapt += 1
    push!(rd.adpNames, name)
    push!(rd.adpLower, lower)
    push!(rd.adpUpper, upper)
    push!(rd.adpCat,   cat)
    push!(rd.adpPolicy, policy)
    push!(rd.adpStage, stage)
    push!(rd.adpDependsOn, depends_on)
    return AdaptiveVariable(m, rd.numAdapt)
end

macro defAdaptVar(args...)
    length(args) <= 1 &&
        error("in @defAdaptVar: expected model as first argument, then variable information.")
    m = esc(args[1])
    x = args[2]
    extra = vcat(args[3:end]...)

    # Identify the variable bounds. Five (legal) possibilities are "x >= lb",
    # "x <= ub", "lb <= x <= ub", or just plain "x"
    if isexpr(x,:comparison)
        # We have some bounds
        if x.args[2] == :>= || x.args[2] == :≥
            if length(x.args) == 5
                # ub >= x >= lb
                x.args[4] == :>= || x.args[4] == :≥ || error("Invalid variable bounds")
                var = x.args[3]
                lb = esc_nonconstant(x.args[5])
                ub = esc_nonconstant(x.args[1])
            else
                # x >= lb
                var = x.args[1]
                @assert length(x.args) == 3
                lb = esc_nonconstant(x.args[3])
                ub = Inf
            end
        elseif x.args[2] == :<= || x.args[2] == :≤
            if length(x.args) == 5
                # lb <= x <= u
                var = x.args[3]
                (x.args[4] != :<= && x.args[4] != :≤) &&
                    error("in @defAdaptVar ($var): expected <= operator after variable name.")
                lb = esc_nonconstant(x.args[1])
                ub = esc_nonconstant(x.args[5])
            else
                # x <= ub
                var = x.args[1]
                # NB: May also be lb <= x, which we do not support
                #     We handle this later in the macro
                @assert length(x.args) == 3
                ub = esc_nonconstant(x.args[3])
                lb = -Inf
            end
        else
            # Its a comparsion, but not using <= ... <=
            error("in @defAdaptVar ($(string(x))): use the form lb <= ... <= ub.")
        end
    else
        # No bounds provided - free variable
        # If it isn't, e.g. something odd like f(x), we'll handle later
        var = x
        lb = -Inf
        ub = Inf
    end

    # separate out keyword arguments
    kwargs = filter(ex->isexpr(ex,:kw), extra)
    extra = filter(ex->!isexpr(ex,:kw), extra)

    # process keyword arguments
    varcat = :Cont
    policy = :Static
    stage  = 0 
    depends_on = Uncertain[]
    for ex in kwargs
        if ex.args[1] == :policy
            policy = esc(ex.args[2])
        elseif ex.args[1] == :stage
            stage = esc(ex.args[2])
        elseif ex.args[1] == :depends_on
            depends_on = esc(ex.args[2])
        else
            error("in @defAdaptVar ($var): Unrecognized keyword argument $(ex.args[1])")
        end
    end

    # Determine variable type (if present).
    # Types: default is continuous (reals)
    if length(extra) > 0
        if extra[1] in [:Bin, :Int]
            gottype = true
            varcat = extra[1]
        end

        if t == :Bin
            if (lb != -Inf || ub != Inf) && !(lb == 0.0 && ub == 1.0)
            error("in @defAdaptVar ($var): bounds other than [0, 1] may not be specified for binary variables.\nThese are always taken to have a lower bound of 0 and upper bound of 1.")
            else
                lb = 0.0
                ub = 1.0
            end
        end

        !gottype && error("in @defAdaptVar ($var): syntax error")
    end

    if isa(var,Symbol)
        # Easy case - a single variable
        return assert_validmodel(m, quote
            $(esc(var)) = 
                AdaptiveVariable($m, $(utf8(string(var))),
                                    $lb, $ub, $(quot(varcat)),
                                    $(quot(policy)), $stage, $(depends_on))
            #registervar($m, $(quot(var)), $(esc(var)))
        end)
    end
    isa(var,Expr) || error("in @defAdaptVar: expected $var to be a variable name")

    error()
end

typealias BothVar Union{Variable,AdaptiveVariable}
typealias VarAffExpr GenericAffExpr{Float64,BothVar}
(*)(c::Number, x::AdaptiveVariable) = VarAffExpr(BothVar[x],Float64[c],0.0)
(+)(a::AffExpr, va::VarAffExpr) = VarAffExpr(vcat(a.vars, va.vars),
                                             vcat(a.coeffs,va.coeffs),
                                             a.constant + va.constant)
# `+` has no method matching +(::JuMP.GenericAffExpr{Float64,Union{JuMP.Variable,JuMPeR.AdaptiveVariable}}, ::JuMP.GenericAffExpr{Float64,JuMPeR.Uncertain})

function JuMP.constructconstraint!(vaff::VarAffExpr, sense::Symbol)
    offset = vaff.constant
    vaff.constant = 0.0
    if sense == :(<=) || sense == :≤
        return VarAffConstraint(vaff, -Inf, -offset)
    elseif sense == :(>=) || sense == :≥
        return VarAffConstraint(vaff, -offset, Inf)
    elseif sense == :(==)
        return VarAffConstraint(vaff, -offset, -offset)
    else
        error("Cannot handle ranged constraint")
    end
end

typealias VarAffConstraint GenericRangeConstraint{VarAffExpr}
addConstraint(m::Model, c::VarAffConstraint) = push!(getRobust(m).varaffcons, c)