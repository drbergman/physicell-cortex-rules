using CSV, DataFrames

pwd()
df = CSV.read("./PhysiCellAUD/config/cells.csv", DataFrame)

d = 410 # desired distance between walls
# ind = abs.(df[!, :x]) .< 0.5*d - 8 # go 8 microns within these walls
ind = abs.(df[!, :x]) .< 0.5*d

new_df = df[ind, :]

CSV.write("./PhysiCellAUD/config/cells_updated.csv", new_df)

using Plots
plot(new_df[!, :x], new_df[!, :y], seriestype = :scatter)