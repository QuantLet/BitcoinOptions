import os
import pickle


def save_dict(moneyness, gdom, gstar, spd, curr_day, tau, maturitydate_char, fname):
    """
    gdom is spd domain
    gstar is estimated hd
    curr_day is current day
    tau is time until maturity
    """

    if os.path.isfile(fname):
        _dict = load_dict(fname)
    else:
        _dict = {}

    curr_key = curr_day + '_' + str(tau) + '_' + str(maturitydate_char)
    _dict[curr_key] =  (moneyness, gdom, gstar, spd)
    with open(fname, 'wb') as f:
        pickle.dump(_dict, f)
    return _dict

def load_dict(fname):
    with open(fname, 'rb') as f:
        _dict = pickle.load(f)
    return _dict


def load_estimated_densities(fname = 'out/real_estimated_densities.csv', allowed_taurange = [0.01, 0.02], only_taus = False):



    x, y, z, s = [], [], [], []
    date, tau, maturitydate = [], [], []
    pk = []

    if os.path.isfile(fname):
        spds = load_dict(fname)

        for key, (moneyness, gdom, gstar, spd) in spds.items():
                try:
                    ysplit = str(key).split('_')
                    currtau = ysplit[1]

                    if len(allowed_taurange) > 0:
                        
                        if float(currtau) >= allowed_taurange[0] and float(currtau) <= allowed_taurange[1]:

                            if max(spd/gstar) < 6:

                                # Extract Date and Tau from Dict Key
                                date.append(ysplit[0]) # Day when instrument is traded
                                tau.append(currtau) # Time to maturity as Decimal
                                maturitydate.append(ysplit[2]) # Maturitydate of Instrument

                                x.append(moneyness)
                                y.append(key)
                                z.append(gstar)
                                s.append(spd)
                                pk.append(spd/gstar)

                                print('loading tau: ', currtau)
                                
                except Exception as e:
                    print(e)
                    print('next')

    if not only_taus:
        return x, z, s, pk, date, tau, maturitydate

    else:
        return tau

