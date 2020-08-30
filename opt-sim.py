"""This file brings in historic, parcel-level loading data from external files, determines when the load from each
parcel arrives at the bay from the sum of 'year built' plus travel time, and establishes a list of decision variables,
which, for each zone, is a list of years during the planning horizon for which some percentage of the parcels in that
zone can be upgraded.

This info serves as input to the function capeN2 which is called repeatedly by the BORG MOEA which different decision
variables as arguments.

There is also an option not to optimize, which you can set in the pars file. If not optimizing, different values can be
input for scenario testing at the end of this file, see exsol2."""

# author: amy piscopo
# nate added "other technologies" into this
# changes from opt-sim12 include replacing choicemat

from borg import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import math
import pars_original as pars
import time
import sys
import openpyxl
from mpl_toolkits.mplot3d.axes3d import Axes3D # this (a gauche) is needed even though PyCharm doesn't turn it orange


"""read in original (unzoned) data for baseline"""  # this needs to be updated with new travel times
df_hist = pd.read_excel(pars.filename, pars.sheetname) # , sep=',', usecols = pars.fields)
print("nparcels is", len(df_hist))

"""for historic load, calculate year first at bay for all parcels and on a zoned basis"""
year_1statbay_unz = df_hist.YEAR_BUILT + df_hist.travelTime  # unz = unzoned (i.e. this data is parcel by parcel)

"""parameters for zones"""
# zone for other technologies, appended to end of the system upgrade params
otech1_size = []  # tech limits in kg/year
otech1_tt = []  # time 'til getting a reduction at estuary
otech1_cost = []  # in per kg/yr terms

syst_tot = np.append(pars.syst_tot, np.full(len(otech1_size), 100))  # otech in % terms
load_tot = np.append(pars.load_tot, otech1_size)
trvtime_avg = np.append(pars.trvtime_avg, otech1_tt)
load_avg = np.true_divide(load_tot, syst_tot)
cost_total_zone = np.append(pars.cost_total_zone, np.multiply(otech1_cost, otech1_size))
cost_per_upg = np.true_divide(cost_total_zone, syst_tot)

"""planning horizon and zone info"""
nzones = len(syst_tot)
xdis = 100
cost_per_upg_div = [x * pars.yrincrm for x in cost_per_upg]

"""simulation parameters"""
plhzndiv = int(pars.planhzn / pars.yrincrm)
reduct_factor = np.append(pars.reduct_factor, np.ones(len(otech1_size)))  # %N remvl w adv septics
costhzn = pars.yr_end - pars.yr_planhzn

"""optimization parameters - in par file too"""
nvars = nzones * plhzndiv # + 1  # +1 is for the aquaculture dv
print("nvars is", nvars)

"""make matrix of upgrade choices in groups of x upgrades at a time"""  # need to make more flexible to "system total"
nchoix = np.zeros(nzones)
truncmax = np.zeros(nzones)  # trunc for truncated
for z in range(nzones):
    nchoix[z] = xdis
    truncmax[z] = syst_tot[z]

# choicemat = np.zeros([nvars - 1, int(xdis + 1)])  # -1 for aquaculture dv
# for z in range(nzones):
#    for y in range(0, plhzndiv):
        # fill_in = np.linspace(0, truncmax[z], xdis+1, endpoint=True)
#        fill_in = np.linspace(0, 100, xdis+1, endpoint=True)
#        for l in range(len(fill_in)):
#            choicemat[z * plhzndiv + y, l] = fill_in[l]


def capeN2(*vars):
    """function evaluation for each set of decision variables"""
    cons1 = 0  # initialized witih unviolated constraint
    cons2 = 37000  # initialize with violated constraint - peak annual N load in "do-nothing" scenario
    # cons3 = pars.yr_end - pars.yr_planhzn  # initialize with violated constraint - TMDL not achieved for entire plhzn
    cons3 = 0 # initialized witih unviolated constraint
    obj_cost = nzones * costhzn * sum(syst_tot) * cost_per_upg[0]  # initialize with highest possible cost
    obj_time = 0
    # obj_time2 = 5000

    load_reduced_byr_aquac = pars.n_aquac * pars.acresaquac  # vars[0] (a placeholder for the acres of aquaculture)

    """set years from vars for when N reaches bay after upgrade"""
    upgrs = np.zeros(nvars)
    zoneupgrs = np.zeros(nzones)
    for z in range(nzones):
        v = 0 + z * plhzndiv
        remaining = truncmax[z]
        for y in range(plhzndiv):
            # print("year %d, remaining %f, vars[v] is %f" % (y, remaining, vars[v]))
            if remaining > 0:
                # upgrs[v] = math.trunc(remaining * math.trunc(vars[v+1]) / 100)
                upgrs[v] = math.trunc(remaining * math.trunc(vars[v]) / 100)
                remaining -= upgrs[v]
                zoneupgrs[z] += upgrs[v]
                v += 1
            else:
                print("exceeded upgrades of %f" % truncmax[z])
                break

    upgrs_valid = upgrs
    # print("upgrs_valid is", np.reshape(upgrs_valid, (nzones, plhzndiv)))

    """determine the year the updated load first hits the bay and the corresponding # upgs"""
    # yr_upd_1statbay = (pars.yr_end + 3000) * np.ones(nvars - 1)  # initialize w large year so won't count as reaching bay
    # nb_upd_1statbay = np.zeros(nvars - 1)  # nb stands for number of septic upgrades
    yr_upd_1statbay = (pars.yr_end + 3000) * np.ones(nvars)  # initialize w large year so won't count as reaching bay
    nb_upd_1statbay = np.zeros(nvars)  # nb stands for number of septic upgrades
    for z in range(nzones):
        for p in range(0, plhzndiv):
            if upgrs_valid[z * plhzndiv + p] > 0:
                yr_upd_1statbay[z * plhzndiv + p] = int(pars.yr_planhzn + trvtime_avg[z] + p * pars.yrincrm)
                nb_upd_1statbay[z * plhzndiv + p] = np.round(upgrs_valid[z * plhzndiv + p])

    """loop through present time to future time to get load curve w upgrades"""
    k = 0
    counttrue = 0
    areaabovetmdl = 0
    load_reduced_total = 0
    sumslug2 = np.zeros(nzones)
    load_actual = np.zeros(int((pars.yr_end - pars.yr_planhzn)/pars.yrincrm))
    yrsachieved = []

    for yy in range(pars.yr_planhzn, pars.yr_end, pars.yrincrm):
        y_loop = yy
        load_reduced_byr = 0
        vec_year = np.ones(nzones * plhzndiv) * y_loop

        slug1 = year_1statbay_unz < y_loop  # this represents the original slug (do-nothing scenario) T if at bay
        lslug1 = np.multiply(slug1.astype(int), df_hist.into_ES_before)
        slug2 = yr_upd_1statbay < vec_year
        nslug2 = np.multiply(slug2.astype(int), nb_upd_1statbay)

        for n in range(nzones):
            indexstart = (n * plhzndiv)
            indexend = k + (n * plhzndiv)
            sumslug2[n] = sum(nslug2[indexstart: indexend])
            load_reduced_byz = load_avg[n] * reduct_factor[n] * sumslug2[n]
            load_reduced_byr += load_reduced_byz

        nupgrs_cuml = sum(sumslug2)  # note: counts upgrades including if year upgrade occurred has already passed

        load_actual[int((yy - pars.yr_planhzn)/pars.yrincrm)] = sum(lslug1) - load_reduced_byr - load_reduced_byr_aquac
        load_reduced_total += load_reduced_byr
        counttrue += np.sum(nslug2)  # not used at present, but counts years target reached by yr-zone
        if load_actual[int((yy - pars.yr_planhzn)/pars.yrincrm)] - pars.TMDL > 0:
            areaabovetmdl += load_actual[int((yy - pars.yr_planhzn)/pars.yrincrm)] - pars.TMDL
        if k < plhzndiv:
            k += 1

        if load_actual[int((yy - pars.yr_planhzn)/pars.yrincrm)] < pars.TMDL:
            yrsachieved = np.append(yrsachieved, yy)
            cons2 = 0  # the TMDL has been met at least once
            obj_time = len(yrsachieved) * pars.yrincrm
            obj_time2 = yrsachieved[0]

    # if np.any(yrsachieved) is False:
    if sum(yrsachieved) == 0:  # ie TMDL is not met ever
        print("solution exceeded time limit, year = ", pars.yr_end)
        cons2 = load_actual[k - 1] - pars.TMDL  # constraint value gets assigned, equal to distance from meeting TMDL
        cons3 = pars.yr_end - pars.yr_planhzn
        obj_time = 0
        # obj_time2 = 5000

    else:  # new constraint - keep load below TMDL
        # tol_achieved = pars.yrincrm  # adjust as desired, this is the amt of 'wiggle room' in meeting the TMDL
        now_yrs_left = pars.yr_end - yrsachieved[0]
        if len(yrsachieved) * pars.yrincrm < now_yrs_left:  # - tol_achieved:
            cons3 = now_yrs_left - len(yrsachieved) * pars.yrincrm

    """calculate cost for each year that the upgrades are in place during the planning horizon"""
    upgrs_appended = []
    multp = np.arange(nzones, dtype=np.int8)
    for n in range(nzones):
        # for each zone, stacks the # of upgrades each year into one long vector
        upgrs_valid_cuml = np.cumsum(upgrs_valid[0 + n * plhzndiv: plhzndiv + n * plhzndiv])
        upgrs_appended = np.append(upgrs_appended, upgrs_valid_cuml)

    vecindex = np.zeros((nzones, plhzndiv), dtype=int)  # changed from planhzn
    for pyr in range(int(plhzndiv)):
        vecindex[:, pyr] = [x * plhzndiv + pyr for x in multp]  # e.g. [2 32 62 92 122]

    """costs calculated here"""
    plhyrs = 0  # index for planning horizon years
    cost_upgs = 0
    cost_upgs2_d = np.zeros((nzones, int(costhzn/pars.yrincrm)))  # this yearly info is off for optimization version
    cost_upgs2_ud = np.zeros((nzones, int(costhzn/pars.yrincrm)))

    if pars.optimize is True:  # then only track cumulative costs using either discounting or no discounting

        for yrs in range(int((pars.yr_end - pars.yr_planhzn) / pars.yrincrm)):
            if pars.discounting_on is False:
                cost_upgs += np.multiply(upgrs_appended[[vecindex[:, plhyrs]]], cost_per_upg_div)  # vec of zones
                # cost_upgs += vars_upgrs_appended[vecindex[n, pyrs]] * cost_per_upg_div[n]
            else:  # ie otherwise, use discounting
                cost_upgs += np.multiply(upgrs_appended[vecindex[:, plhyrs]], cost_per_upg_div) / np.power(1 + pars.discountrate, yrs * pars.yrincrm)
            if plhyrs < plhzndiv-1:
                plhyrs += 1

        """assign calculated cost to objective function for borg"""
        if cons1 > 0:
            obj_cost += (cons1 * 10000)  # penalize the objective function to train the algorithm
        else:
            obj_cost = sum(cost_upgs)  # sums across the zones

    else:  # since no optimization, save yearly costs, discounted & not discounted
        obj_cost = 0
        for yrs in range(int(costhzn/pars.yrincrm)):
            cost_upgs2_ud[:, yrs] = np.multiply(upgrs_appended[[vecindex[:, plhyrs]]], cost_per_upg_div)
            print("(cost_upgs2_ud)", (cost_upgs2_ud))
            # cost_upgs2_ud += np.multiply(upgrs_appended[[vecindex[:, plhyrs]]], cost_per_upg_div)  # vec of zones
            # print("in yrs loop", cost_upgs2_ud)
            # time.sleep(2)
            cost_upgs2_d[:, yrs] = np.multiply(upgrs_appended[vecindex[:, plhyrs]], cost_per_upg_div) / np.power(1 + pars.discountrate, yrs * pars.yrincrm)
            print("len(cost_upgs2_d)", len(cost_upgs2_d))
            # cost_upgs2_d += np.multiply(upgrs_appended[vecindex[:, plhyrs]], cost_per_upg_div) / np.power(1 + pars.discountrate, yrs)

            if plhyrs < plhzndiv-1:
                plhyrs += 1

    if pars.show_historic_plot is True:
        """loop through historic time to future time to get load curve assuming no upgrs, using parcel by parcel data"""
        w_totmassperyear = np.zeros(pars.yr_end - pars.yr_spinstart)  # yrincrm not used here bc historic data is hourly
        year_i = pars.yr_spinstart
        for i in range(len(w_totmassperyear)):
            vecyear_s = np.ones(len(df_hist)) * year_i
            slug = year_1statbay_unz < vecyear_s
            w_totmassperyear[i] = sum(df_hist.into_ES_before * slug.astype(int))
            year_i += 1

        if pars.optimize is False:
            """plot N load to bay with and without septic upgrades"""
            pl.figure(1)
            pyearsspin = np.arange(pars.yr_spinstart, pars.yr_end, 1)
            pyearshzn = np.arange(pars.yr_planhzn, pars.yr_end, pars.yrincrm)
            pl.plot(pyearsspin, w_totmassperyear, label='w/o plan')
            pl.plot(pyearshzn, load_actual, label='w plan')
            pl.plot(pyearsspin, pars.TMDL * np.ones(len(pyearsspin)), 'r-', label='TDML')
            pl.xlabel('years')
            pl.ylabel('N loading to bay')
            x1, x2, y1, y2 = pl.axis()
            pl.axis((pars.yr_spinstart, pars.yr_end, y1, y2))
            pl.legend(loc=4)

            """plot costs over time"""
            pl.figure(2)
            # axis=0 sums along the rows (zones) producing column (yearly) totals
            print("pyearshzn", pyearshzn)
            pl.plot(pyearshzn, np.true_divide(np.sum(cost_upgs2_ud, axis=0), 1e6), label='not discounted', color='g')
            pl.plot(pyearshzn, np.true_divide(np.sum(cost_upgs2_d, axis=0), 1e6), '--', label='discounted',  color='g')
            pl.xlabel('years')
            pl.ylabel('million $')
            x1, x2, y1, y2 = pl.axis()
            pl.axis((pars.yr_planhzn, 2160 , y1, y2))
            pl.legend(loc=4)
            pl.show()

    objs = [-obj_time, obj_cost, areaabovetmdl] #, obj_time2]
    cons = [cons1, cons2, cons3]
    return objs, cons, upgrs_appended, cost_upgs2_d, cost_upgs2_ud

# seed loop here
for i in range(pars.nseeds):
    print("seed index", i)

    if pars.optimize is True:
        """then call Borg to run optimization"""
        borg = Borg(nvars, pars.nobjs, pars.ncons, capeN2)
        borg.setEpsilons(*[1, 12000, 1000])
        Configuration.seed(value = i)
        #if pars.acresaquac > 0:
        #    borg.setBounds([0, pars.acresaquac], *[[0, 100]] * (nvars-1))  # %0-100 for each of the zone-year decisions
        #else:
        #    borg.setBounds(*[[0, 100]] * nvars)
        borg.setBounds(*[[0, 100]] * nvars)

        result = borg.solve({"maxEvaluations": pars.NFE})  # NFE
        print("pars.NFE is", pars.NFE)
        nsol = 0
        for solution in result:
            print(solution.getObjectives())
            nsol += 1

        vec_totyrsmet = []
        vec_cost = []
        vec_areatmdl = []
        fileout = open("optimset_seed" + str(i) + ".txt", 'w')
        fileout.write('%d\n' % nsol)
        for solution in result:
            objslist = solution.getObjectives()
            fileout.write('%.1f\t%.1f\t%.1f\n' % (objslist[0], objslist[1], objslist[2]))
            vec_totyrsmet[len(vec_totyrsmet):] = [-objslist[0]]
            vec_cost[len(vec_cost):] = [objslist[1] / 1000000]
            vec_areatmdl[len(vec_areatmdl):] = [objslist[2]]

        for solution in result:
            dvslist = solution.getVariables()
            for i in range(0, len(dvslist)):  # 0 to indicate that aquac is printed
                fileout.write('%d\t' % dvslist[i])
            fileout.write('\n')
        fileout.close()

        if pars.nseeds < 2:  # don't plot figure at the end of optimization unless single seed
            fig = pl.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(vec_totyrsmet, vec_cost, vec_areatmdl, c='b') #, marker ='o', markersize=4)
            ax.set_xlabel('years TMDL is met')
            ax.set_ylabel('cost (1M USD)')
            ax.set_zlabel('env damage function (kg)')
            pl.savefig('objs.png')
            pl.show()

    else:
        """if 'optimize' is False, don't optimize, run simulation of single scenario"""
        exsol2 = np.zeros(nvars)
        # fills in the same gradual choice each year for in each zone, "1+" because of aquaculture being the first choice
        for yr in range(plhzndiv):
            exsol2[yr] = 20
            exsol2[yr  + plhzndiv * 1] = 20
            exsol2[yr  + plhzndiv * 2] = 20
            exsol2[yr  + plhzndiv * 3] = 20
            exsol2[yr  + plhzndiv * 4] = 20
            exsol2[yr  + plhzndiv * 5] = 0
            exsol2[yr  + plhzndiv * 6] = 0

            #exsol2 = [
            #     0,  0, 97, 0,  0, 3, 23, 54, 18, 12,  2, 19,  5,  0, 94,
            #     0,  1,  0, 0,  3, 0,  0,  1,  1,  0,  0,  1,  3,  0,  0,
            #     0,  0,  1, 0,  0, 0,  1, 39, 14, 14, 15,  1, 14, 12,  3,
            #     0,  0,  1, 1,  1, 1, 18,  5, 17,  0, 40,  9, 61,  4,  3,
            #     0,  0,  3, 0,  0, 0,  0,  0, 19,  0,  0,  6, 18, 87, 41,
            #     0, 29, 17, 4,	0, 2, 26,  9,  0,  0,  2,  0,  0,  0,  0,
            #     0,  0,  0, 0,	0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0]

            #exsol2 = [
            #     0, 0, 97,
            #     0, 1, 0,
            #     0, 0, 1,
            #     0, 0, 1,
            #     0, 0, 3,
            #     0, 29, 17,
            #     0, 0, 0]

        print("exsol2 is", exsol2)
        vars = exsol2

        [a, b, c, d, e] = capeN2(*exsol2)
        print("upgrs_appended", c)
        print("discounted costs", d)
        print("regular costs", e)