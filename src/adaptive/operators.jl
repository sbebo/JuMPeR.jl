#-----------------------------------------------------------------------
# JuMPeR  --  JuMP Extension for Robust Optimization
# http://github.com/IainNZ/JuMPeR.jl
#-----------------------------------------------------------------------
# Copyright (c) 2015: Iain Dunning
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#-----------------------------------------------------------------------
# src/adaptive/macro.jl
# Adaptive robust optimization support - operators
#-----------------------------------------------------------------------

(*)(c::Number, x::Adaptive) = AdaptAffExpr(Adaptive[x], Float64[c], 0.0)

(+)(a::AffExpr, b::AdaptAffExpr) = AdaptAffExpr(vcat(a.vars,   b.vars),
                                                vcat(a.coeffs, b.coeffs),
                                                a.constant + b.constant)

(+)(a::Adaptive, b::Adaptive) = AdaptAffExpr(Adaptive[a,b], Float64[1,1], 0.0)

(+)(a::AdaptAffExpr, b::UncExpr) = UncAffExpr(copy(a.vars),
                                              map(UncExpr,a.coeffs),
                                              copy(b))

+(a::UncAffExpr, b::AdaptAffExpr) = UncAffExpr(vcat(a.vars, b.vars),
                                               vcat(a.coeffs, map(UncExpr,b.coeffs)),
                                               a.constant + b.constant)