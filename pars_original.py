"""parameter file"""

"""run type parameters"""
optimize = True
show_historic_plot = True    # if doing more than a demo optimization, this should be false

"""optimization parameters"""
nobjs        = 3             # maximize years that the TMDL is reached, minimize cost, and minimize time to first reach TMDL
ncons        = 3             # chosen upgrades exceed maximum possible upgrades, solution doesn't achieve reduction target
nseeds       = 1
NFE          = 500

"""external files with loading info"""
filename     = 'Three_bays_MVP_time_yearbuilt_simple3.xlsx'  # '3Bays_historic.xlsx'
sheetname    = 'Historic groups'
fields       = ['into_ES_before', 'travelTime', 'YEAR_BUILT']

"""aquaculture parameters"""
acresaquac   = 0             # the amount of aquaculture implemented - 0 to turn off aquaculture
n_aquac      = 113.4         # removal of N by aquaculture per acre per year (in kg)

"""cost parameters"""
discounting_on = False         # note that both present values and discounted costs are calculated when optimize is False
discountrate = 0.05

"""time parameters"""
yr_spinstart = 1940          # used for the historical load
yr_planhzn   = 2020          # the first year of the planning horizon (when upgrades can start to occur)
yr_end       = 2170

"""system parameters"""
planhzn	     = 50            # duration of planning horizon (in years)
nzones       = 7             # number of zones (each zones has its own travel time) - must equal len(syst_tot)
yrincrm      = 5             # year increment (i.e. choice variable to upgrade occurs every 'yrincrm' years)
xupg         = 1             # upgrade increment (i.e. upgrades can occur in units of 'xupg')
TMDL         = 11506.4       # not the actual TMDL but the target to reach based on controllable sources of N

syst_tot = [3544, 1149, 514, 178, 119, 241, 451]
load_tot = [19799.78, 5198.48, 2199.35, 712.64, 593.60, 1189.21, 2391.40]
trvtime_avg = [5, 15, 25, 35, 45, 75, 100]
cost_total_zone = [46839289.08, 13582058.46, 5744132.101, 2175138.579, 1568196.434, 2850561.803, 5270299.302]
reduct_factor = [.81, .81, .81, .81, .81, .81, .81]


