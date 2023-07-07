using Statistics
using QuantumCumulants
using ModelingToolkit, OrdinaryDiffEq
using Distributions
using Plots

function tensor_cavs(N)
    Hlist = [FockSpace(Symbol(:cavity,i)) for i=1:N]
    return ⊗(Hlist...)
end

N=6
orderc = 1
h = tensor_cavs(N)
a(i) = Destroy(h, Symbol(:a, i), i)
a_dag(i) = Create(h, Symbol(:a, i), i)
@cnumbers Δ γ J U
H = sum(U*a_dag(i)*a_dag(i)*a(i)*a(i) -Δ*a_dag(i)*a(i) for i=1:N) + sum(J*a_dag(i)*a(i+1) + J*a_dag(i+1)*a(i) for i=1:N-1) + J*(a_dag(1)*a(N) + a_dag(N)*a(1))
D = [a(i) for i=1:N]
rates = [γ for i=1:N]
H

#cumulant equations
eqs = meanfield([a(N)], H, D; rates=rates, order=orderc)
eqs_completed = complete(eqs)

#Numerical resolution of the mean-field equations
using ModelingToolkit, OrdinaryDiffEq
@named sys = ODESystem(eqs_completed)
p0 = (Δ=>-4, J=>1, γ=>1, U=>0.0)

list = [2, -1, 1im, 2im,2+1im,-1-1im]
l = length(list)

u0 = zeros(ComplexF64, length(eqs_completed))

for i in 1:Int(N)
 #  num = mod(i, 4)
  #  if(num == 0)
  #      num=4
  #  end
    #u0[i] = list[num]
    u0[i] = list[i]
end
print(u0)

#solve ODEs
t_lim = 1.0
prob = ODEProblem(sys,u0,(0.0,t_lim),p0)
sol = solve(prob,Vern7(), dense=false, saveat = collect(range(0,t_lim,step=0.01))  )
print("ciao")
print(p0)
print(u0)

# plot alphas
plot(sol.t, real.(sol[a(1)]), xlabel="t", ylabel="a", label="real cav 1")
plot!(sol.t, real.(sol[a(2)]), xlabel="t", ylabel="a", label="real cav 2")
plot!(sol.t, real.(sol[a(3)]), xlabel="t", ylabel="a", label="real cav 3")
plot!(sol.t, real.(sol[a(4)]), xlabel="t", ylabel="a", label="real cav 4")
plot!(sol.t, real.(sol[a(5)]), xlabel="t", ylabel="a", label="real cav 5")

#plot n
n = abs2.(sol[a(1)])
plot(sol.t, n, xlabel="t", ylabel="n", label="mean photon density cavity 1")
n = abs2.(sol[a(5)])
plot!(sol.t, n, xlabel="t", ylabel="n", label="mean photon density cavity 5")

# save
using NPZ
rr= mapreduce(permutedims, vcat, sol.u)
npzwrite("6mode_julia_U0p1_1storder.npz", Dict("u" => rr, "t" => sol.t))