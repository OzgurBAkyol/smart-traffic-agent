#!/usr/bin/env python3
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2010-2022 German Aerospace Center (DLR) and others.
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# https://www.eclipse.org/legal/epl-2.0/
# This Source Code may also be made available under the following Secondary
# Licenses when the conditions for such availability set forth in the Eclipse
# Public License 2.0 are satisfied: GNU General Public License, version 2
# or later which is available at
# https://www.gnu.org/licenses/old-licenses/gpl-2.0-standalone.html
# SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

# @file    gtfs2pt.py
# @author  Jakob Erdmann
# @author  Michael Behrisch
# @date    2018-08-28

"""
Maps GTFS data to a given network, generating routes, stops and vehicles
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import io
import glob
import subprocess
from collections import defaultdict
import zipfile
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

sys.path += [os.path.join(os.environ["SUMO_HOME"], "tools"), os.path.join(os.environ['SUMO_HOME'], 'tools', 'route')]  # noqa
import route2poly  # noqa
import sumolib  # noqa
import tracemapper  # noqa

import gtfs2fcd  # noqa
import gtfs2osm  # noqa


def get_options(args=None):
    ap = sumolib.options.ArgumentParser()
    ap = gtfs2fcd.add_options()
    # ----------------------- general options ---------------------------------
    ap.add_argument("-n", "--network", fix_path=True, required=True,
                    help="sumo network to use")
    ap.add_argument("--route-output",
                    help="file to write the generated public transport vehicles to")
    ap.add_argument("--additional-output",
                    help="file to write the generated public transport stops and routes to")
    ap.add_argument("--duration", default=10,
                    type=int, help="minimum time to wait on a stop")
    ap.add_argument("--bus-stop-length", default=13, type=float,
                    help="length for a bus stop")
    ap.add_argument("--train-stop-length", default=110, type=float,
                    help="length for a train stop")
    ap.add_argument("--tram-stop-length", default=60, type=float,
                    help="length for a tram stop")
    ap.add_argument("--sort", action="store_true", default=False,
                    help="sorting the output-file")

    # ----------------------- fcd options -------------------------------------
    ap.add_argument("--network-split",
                    help="directory to write generated networks to")
    # ap.add_argument("--network-split.vclass", action="store_true", default=False,
    #                        help="use the allowed vclass instead of the edge type to split the network")
    ap.add_argument("--warn-unmapped", action="store_true", default=False,
                    help="warn about unmapped routes")
    ap.add_argument("--mapperlib", default="lib/fcd-process-chain-2.2.2.jar",
                    help="mapping library to use")
    ap.add_argument("--map-output",
                    help="directory to write the generated mapping files to")
    ap.add_argument("--map-output-config", default="conf/output_configuration_template.xml",
                    help="output configuration template for the mapper library")
    ap.add_argument("--map-input-config", default="conf/input_configuration_template.xml",
                    help="input configuration template for the mapper library")
    ap.add_argument("--map-parameter", default="conf/parameters_template.xml",
                    help="parameter template for the mapper library")
    ap.add_argument("--poly-output", help="file to write the generated polygon files to")
    ap.add_argument("--fill-gaps", default=5000, type=float,
                    help="maximum distance between stops")
    ap.add_argument("--skip-fcd", action="store_true", default=False,
                    help="skip generating fcd data")
    ap.add_argument("--skip-map", action="store_true", default=False,
                    help="skip network mapping")

    # ----------------------- osm options -------------------------------------
    ap.add_argument("--osm-routes", help="osm routes file")
    ap.add_argument("--warning-output",
                    help="file to write the unmapped elements from gtfs")
    ap.add_argument("--dua-repair-output",
                    help="file to write the osm routes with errors")
    ap.add_argument("--bbox",
                    help="define the bounding box to filter the gtfs data, format: W,S,E,N")
    ap.add_argument("--repair", help="repair osm routes", action='store_true')
    ap.add_argument("--min-stops", default=1, type=int,
                    help="minimum number of stops a public transport line must have to be imported")

    options = ap.parse_args(args)

    options = gtfs2fcd.check_options(options)

    if options.additional_output is None:
        options.additional_output = options.region + "_pt_stops.add.xml"
    if options.route_output is None:
        options.route_output = options.region + "_pt_vehicles.add.xml"
    if options.warning_output is None:
        options.warning_output = options.region + "_missing.xml"
    if options.dua_repair_output is None:
        options.dua_repair_output = options.region + "_repair_errors.txt"
    if options.map_output is None:
        options.map_output = os.path.join('output', options.region)
    if options.network_split is None:
        options.network_split = os.path.join('resources', options.region)

    return options


def splitNet(options):
    netcCall = [sumolib.checkBinary("netconvert"), "--no-internal-links", "--numerical-ids", "--no-turnarounds",
                "--offset.disable-normalization", "--output.original-names", "--aggregate-warnings", "1",
                "--junctions.corner-detail", "0", "--dlr-navteq.precision", "0", "--geometry.avoid-overlap", "false"]
    if not os.path.exists(options.network_split):
        os.makedirs(options.network_split)
    numIdNet = os.path.join(options.network_split, "numerical.net.xml")
    if os.path.exists(numIdNet) and os.path.getmtime(numIdNet) > os.path.getmtime(options.network):
        print("Reusing old", numIdNet)
    else:
        subprocess.call(netcCall + ["-s", options.network, "-o", numIdNet,
                                    "--discard-params", "origId,origFrom,origTo"])
    edgeMap = {}
    seenTypes = set()
    for e in sumolib.net.readNet(numIdNet).getEdges():
        edgeMap[e.getID()] = e.getLanes()[0].getParam("origId", e.getID())
        seenTypes.add(e.getType())
    typedNets = {}
    for inp in glob.glob(os.path.join(options.gpsdat, "gpsdat_*.csv")):
        mode = os.path.basename(inp)[7:-4]
        if not options.modes or mode in options.modes.split(","):
            netPrefix = os.path.join(options.network_split, mode)
            sumoType = gtfs2osm.OSM2SUMO_MODES[mode]
            edgeTypes = [sumoType]
            if "rail" in sumoType or sumoType == "subway":
                edgeTypes = ["railway." + sumoType]
            elif sumoType in ("tram", "bus"):
                edgeTypes = ["railway.tram"] if sumoType == "tram" else []
                for hwType in ("bus_guideway", "living_street", "motorway", "motorway_link", "primary", "primary_link",
                               "residential", "secondary", "secondary_link", "tertiary", "tertiary_link",
                               "trunk", "trunk_link", "unclassified", "unsurfaced"):
                    if sumoType == "tram":
                        edgeTypes.append("highway.%s|railway.tram" % hwType)
                    else:
                        edgeTypes.append("highway." + hwType)
            edgeType = ",".join(filter(lambda t: t in seenTypes, edgeTypes))
            if edgeType:
                if (os.path.exists(netPrefix + ".net.xml") and
                        os.path.getmtime(netPrefix + ".net.xml") > os.path.getmtime(numIdNet)):
                    print("Reusing old", netPrefix + ".net.xml")
                else:
                    subprocess.call(netcCall + ["-s", numIdNet, "-o", netPrefix + ".net.xml",
                                                "--dlr-navteq-output", netPrefix,
                                                "--dismiss-vclasses", "--keep-edges.by-type", edgeType])
                typedNets[mode] = (inp, netPrefix)
    return edgeMap, typedNets


def mapFCD(options, typedNets):
    for o in glob.glob(os.path.join(options.map_output, "*.dat")):
        os.remove(o)
    outConf = os.path.join(os.path.dirname(options.map_output_config), "output_configuration.xml")
    with open(options.map_output_config) as inp, open(outConf, "w") as outp:
        outp.write(inp.read() % {"output": options.map_output})
    for railType, (gpsdat, netPrefix) in typedNets.items():
        conf = os.path.join(os.path.dirname(options.map_input_config), "input_configuration_%s.xml") % railType
        with open(options.map_input_config) as inp, open(conf, "w") as outp:
            outp.write(inp.read() % {"input": gpsdat, "net_prefix": netPrefix})
        param = os.path.join(os.path.dirname(options.map_parameter), "parameters_%s.xml") % railType
        with open(options.map_parameter) as inp, open(param, "w") as outp:
            outp.write(inp.read() % {"radius": 100 if railType in ("bus", "tram") else 1000})
        call = "java -mx16000m -jar %s %s %s %s" % (options.mapperlib, conf, outConf, param)
        if options.verbose:
            print(call)
        sys.stdout.flush()
        subprocess.call(call, shell=True)


def traceMap(options, typedNets, radius=100):
    routes = defaultdict(list)
    for railType in typedNets.keys():
        if options.verbose:
            print("mapping", railType)
        net = sumolib.net.readNet(os.path.join(options.network_split, railType + ".net.xml"))
        netBox = net.getBBoxXY()
        numTraces = 0
        filePath = os.path.join(options.fcd, railType + ".fcd.xml")
        if not os.path.exists(filePath):
            return []
        traces = tracemapper.readFCD(filePath, net, True)
        for tid, trace in traces:
            numTraces += 1
            minX, minY, maxX, maxY = sumolib.geomhelper.addToBoundingBox(trace)
            if (minX < netBox[1][0] + radius and minY < netBox[1][1] + radius and
                    maxX > netBox[0][0] - radius and maxY > netBox[0][1] - radius):
                mappedRoute = sumolib.route.mapTrace(trace, net, radius, fillGaps=options.fill_gaps)
                if mappedRoute:
                    routes[tid] = [e.getID() for e in mappedRoute]
        if options.verbose:
            print("mapped", numTraces, "traces to", len(routes), "routes.")
    return routes


def generate_polygons(net, routes, outfile):
    colorgen = sumolib.miscutils.Colorgen(('random', 1, 1))

    class PolyOptions:
        internal = False
        spread = 0.2
        blur = 0
        geo = True
        layer = 100
    with open(outfile, 'w') as outf:
        outf.write('<polygons>\n')
        for vehID, edges in routes.items():
            route2poly.generate_poly(PolyOptions, net, vehID, colorgen(), edges, outf)
        outf.write('</polygons>\n')


def map_stops(options, net, routes, rout):
    stops = defaultdict(list)
    stopDef = set()
    rid = None
    for inp in sorted(glob.glob(os.path.join(options.fcd, "*.fcd.xml"))):
        railType = os.path.basename(inp)[:-8]
        typedNetFile = os.path.join(options.network_split, railType + ".net.xml")
        if not os.path.exists(typedNetFile):
            print("Warning! No net", typedNetFile)
            continue
        if options.verbose:
            print("Reading", typedNetFile)
        typedNet = sumolib.net.readNet(typedNetFile)
        seen = set()
        fixed = set()
        for veh in sumolib.xml.parse_fast(inp, "vehicle", ("id", "x", "y", "until", "name",
                                                           "fareZone", "fareSymbol", "startFare")):
            addAttrs = ' friendlyPos="true" name="%s"' % veh.attr_name
            params = ""
            if veh.fareZone:
                params = "".join(['        <param key="%s" value="%s"/>\n' %
                                  p for p in (('fareZone', veh.fareZone), ('fareSymbol', veh.fareSymbol),
                                              ('startFare', veh.startFare))])
            if rid != veh.id:
                lastIndex = 0
                lastPos = -1
                rid = veh.id
            if rid not in routes:
                if options.warn_unmapped and rid not in seen:
                    print("Warning! Not mapped", rid)
                    seen.add(rid)
                continue
            route = routes[rid]
            if rid not in fixed:
                routeFixed = [route[0]]
                for routeEdgeID in route[1:]:
                    path, _ = typedNet.getShortestPath(typedNet.getEdge(routeFixed[-1]), typedNet.getEdge(routeEdgeID))
                    if path is None or len(path) > options.fill_gaps + 2:
                        error = "no path found" if path is None else "path too long (%s)" % len(path)
                        print("Warning! Disconnected route '%s', %s. Keeping longer part." % (rid, error))
                        if len(routeFixed) > len(route) // 2:
                            break
                        routeFixed = [routeEdgeID]
                    else:
                        if len(path) > 2:
                            print("Warning! Fixed connection", rid, len(path))
                        routeFixed += [e.getID() for e in path[1:]]
                if rid not in routes:
                    continue
                route = routes[rid] = routeFixed
                fixed.add(rid)
            if railType == "bus":
                stopLength = options.bus_stop_length
            elif railType == "tram":
                stopLength = options.tram_stop_length
            else:
                stopLength = options.train_stop_length
            result = gtfs2osm.getBestLane(typedNet, veh.x, veh.y, 200, stopLength,
                                          route[lastIndex:], railType, lastPos)
            if result is None:
                if options.warn_unmapped:
                    print("Warning! No stop for coordinates %.2f, %.2f" % (veh.x, veh.y), "on", veh)
                continue
            laneID, start, end = result
            lane = typedNet.getLane(laneID)
            edgeID = lane.getEdge().getID()
            lastIndex = route.index(edgeID, lastIndex)
            lastPos = end
            origEdgeID = lane.getParam("origId", edgeID)
            origLaneID = "%s_%s" % (origEdgeID, lane.getIndex())
            stop = "%s:%.2f" % (origEdgeID, end)
            if stop not in stopDef:
                stopDef.add(stop)
                typ = "busStop" if railType == "bus" else "trainStop"
                rout.write(u'    <%s id="%s" lane="%s" startPos="%.2f" endPos="%.2f"%s>\n%s' %
                           (typ, stop, origLaneID, start, end, addAttrs, params))
                for a in gtfs2osm.getAccess(net, veh.x, veh.y, 100, origLaneID):
                    rout.write(a)
                rout.write(u'    </%s>\n' % typ)
            stops[rid].append((stop, int(veh.until)))
    return stops


def filter_trips(options, routes, stops, outfile, begin, end):
    numDays = end // 86400
    if end % 86400 != 0:
        numDays += 1
    with io.open(outfile, 'w', encoding="utf8") as outf:
        sumolib.xml.writeHeader(outf, os.path.basename(__file__), "routes", options=options)
        if options.sort:
            vehs = defaultdict(lambda: "")
        for inp in glob.glob(os.path.join(options.fcd, "*.rou.xml")):
            for veh in sumolib.xml.parse_fast_structured(inp, "vehicle", ("id", "route", "type", "depart", "line"),
                                                         {"param": ["key", "value"]}):
                if len(routes.get(veh.route, [])) > 0 and len(stops.get(veh.route, [])) > 1:
                    until = stops[veh.route][0][1]
                    for d in range(numDays):
                        depart = max(0, d * 86400 + int(veh.depart) + until - options.duration)
                        if begin <= depart < end:
                            if d != 0 and veh.id.endswith(".trimmed"):
                                # only add trimmed trips the first day
                                continue
                            line = (u'    <vehicle id="%s.%s" route="%s" type="%s" depart="%s" line="%s">\n' %
                                    (veh.id, d, veh.route, veh.type, depart, veh.line))
                            for p in veh.param:
                                line += u'        <param key="%s" value="%s"/>\n' % p
                            line += u'    </vehicle>\n'
                            if options.sort:
                                vehs[depart] += line
                            else:
                                outf.write(line)
        if options.sort:
            for _, vehs in sorted(vehs.items()):
                outf.write(vehs)
        outf.write(u'</routes>\n')


def main(options):
    if options.verbose:
        print('Loading net')
    net = sumolib.net.readNet(options.network)

    if options.osm_routes:
        # Import PT from GTFS and OSM routes
        if not options.bbox:
            BBoxXY = net.getBBoxXY()
            BBoxLonLat = (net.convertXY2LonLat(BBoxXY[0][0], BBoxXY[0][1]),
                          net.convertXY2LonLat(BBoxXY[1][0], BBoxXY[1][1]))
            options.bbox = (BBoxLonLat[0][0], BBoxLonLat[0][1],
                            BBoxLonLat[1][0], BBoxLonLat[1][1])
        else:
            options.bbox = [float(coord) for coord in options.bbox.split(",")]

        gtfsZip = zipfile.ZipFile(sumolib.open(options.gtfs, False))
        routes, trips_on_day, shapes, stops, stop_times = gtfs2osm.import_gtfs(options, gtfsZip)

        if shapes is None:
            print('Warning: Importing OSM routes currently requires a GTFS file with shapes.')
            options.osm_routes = None
        else:
            (gtfs_data, trip_list,
             filtered_stops,
             shapes, shapes_dict) = gtfs2osm.filter_gtfs(options, routes,
                                                         trips_on_day, shapes,
                                                         stops, stop_times)

            osm_routes = gtfs2osm.import_osm(options, net)

            (mapped_routes, mapped_stops,
             missing_stops, missing_lines) = gtfs2osm.map_gtfs_osm(options, net, osm_routes, gtfs_data, shapes,
                                                                   shapes_dict, filtered_stops)

            gtfs2osm.write_gtfs_osm_outputs(options, mapped_routes, mapped_stops,
                                            missing_stops, missing_lines,
                                            gtfs_data, trip_list, shapes_dict, net)
    if not options.osm_routes:
        # Import PT from GTFS
        if not options.skip_fcd:
            gtfs2fcd.main(options)
        edgeMap, typedNets = splitNet(options)
        if os.path.exists(options.mapperlib):
            if not options.skip_map:
                mapFCD(options, typedNets)
            routes = defaultdict(lambda: [])
            for o in glob.glob(os.path.join(options.map_output, "*.dat")):
                for line in open(o):
                    time, edge, speed, coverage, id, minute_of_week = line.split('\t')[:6]
                    routes[id].append(edge)
        else:
            if not gtfs2fcd.dataAvailable(options):
                sys.exit("No GTFS data found for given date %s." % options.date)
            if options.mapperlib != "tracemapper":
                print("Warning! No mapping library found, falling back to tracemapper.")
            routes = traceMap(options, typedNets)

        if options.poly_output:
            generate_polygons(net, routes, options.poly_output)
        with io.open(options.additional_output, 'w', encoding="utf8") as rout:
            sumolib.xml.writeHeader(rout, os.path.basename(__file__), "additional")
            stops = map_stops(options, net, routes, rout)
            for vehID, edges in routes.items():
                if edges:
                    rout.write(u'    <route id="%s" edges="%s">\n' % (vehID, " ".join([edgeMap[e] for e in edges])))
                    offset = None
                    for stop in stops[vehID]:
                        if offset is None:
                            offset = stop[1]
                        rout.write(u'        <stop busStop="%s" duration="%s" until="%s"/>\n' %
                                   (stop[0], options.duration, stop[1] - offset))
                    rout.write(u'    </route>\n')
                else:
                    print("Warning! Empty route", vehID)
            rout.write(u'</additional>\n')
        filter_trips(options, routes, stops, options.route_output, options.begin, options.end)


if __name__ == "__main__":
    main(get_options())
