STRICT = {"PoDCRouter.minContribution": "0.01", "PoDCRouter.explore": "0.1"}
PROMO = {"SybilPoDCRouter.fakeAckInterval": "10"}


def att(router, extra=None):
    e = {"Scenario.nrofHostGroups": "7", "Group7.groupID": "x", "Group7.nrofHosts": "10",
         "Group7.router": router, "Events1.hosts": "0,100", "DataStatsReport.maliciousPrefix": "x"}
    e.update(extra or {})
    return e
