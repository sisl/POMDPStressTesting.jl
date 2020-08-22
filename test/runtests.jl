using Test

testfile2trace = ["Walk1D.jl"=>[0x55e61439, 0x5e1cdeab, 0xae0d4f5c, 0xf009ca94, 0x6e6cdbe7, 0x9d92eb62, 0x7bd81654, 0xf92ccb6a, 0x4b606e85, 0x7f7f7b4f]]

testdir = joinpath(dirname(@__DIR__), "test")

cd(testdir) do
    @testset "examples" begin
        for (testfile, trace) in testfile2trace
            include(testfile)
            @test map(a->a.seed, action_trace) == trace
        end
    end
end