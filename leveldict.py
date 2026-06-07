import re


class Transition:
    def __init__(self, transition):
        self.iso1 = int(transition[1])
        self.i1 = int(transition[2])
        self.iso2 = int(transition[3])
        self.i2 = int(transition[4])
        self.de = None
        self.g1 = None
        self.g2 = None


class PhotIon(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.a0 = float(transition[5])
        self.b0 = float(transition[6])
        self.c0 = float(transition[7])
        self.d0 = 0.0
        if len(transition) > 8:
            self.d0 = float(transition[8])

class Phis(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.a0 = float(transition[5])
        self.a1 = float(transition[6])
        self.a2 = float(transition[7])
        self.a3 = float(transition[8])
        self.n = float(transition[9])
        self.de = float(transition[10])
        self.emin = float(transition[11])
        self.emax = float(transition[12])
    


class Phxs(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.f = float(transition[5])
        self.wavelength = float(transition[6])
        self.width = 0.0
        if len(transition) > 7:
            self.width = float(transition[7])


class Colex(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.c0 = float(transition[5])
        self.c1 = float(transition[6])
        self.c2 = float(transition[7])
        self.c3 = float(transition[8])
        self.a0 = float(transition[9])
        self.c4 = 0.0
        if len(transition) > 10:
            self.a0 = float(transition[10])


class Augxs(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.A = float(transition[5])


class Colon(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.c0 = float(transition[5])
        self.c1 = float(transition[6])
        self.c2 = float(transition[7])
        self.c3 = float(transition[8])
        self.a0 = 0.0
        self.c4 = 0.0
        if len(transition) > 9:
            self.a0 = float(transition[9])
        if len(transition) > 10:
            self.c4 = float(transition[10])


class Sampson(Transition):
    def __init__(self, transition):
        transition = transition.strip().split()
        super().__init__(transition)
        self.c0 = float(transition[5])
        self.c1 = float(transition[6])
        self.c2 = float(transition[7])
        self.c3 = float(transition[8])
        self.s0 = float(transition[9])


class Level:
    def __init__(self, level, iso_name=None, iso_energy=None):
        level = level.strip().split()
        self.iso = int(level[1])
        self.i = int(level[2])
        self.name = level[3]
        self.g = float(level[4])
        self.energy = float(level[5])
        self.config = [int(i) for i in level[6:-1]]
        self.nmax = int(level[-1])
        self.iso_name = iso_name
        self.iso_energy = iso_energy
        self.index = None
        self.config_string = self.set_config_string()

    def set_index(self, index):
        self.index = index

    def set_config_string(self):
        prefix = ["1s", "2s", "2p", "3s", "3p", "3d", "4s", "4p"]
        return " ".join("%s%d" % (s, e) for s, e in zip(prefix, self.config) if e > 0) + " (%d)" % self.g


class LevelDB:

    def __init__(self, datafile_name):
        self.datafile_name = datafile_name
        self.levels = list()
        self._statical_weight_lookup = dict()
        self._iso_energy_lookup = dict()
        self._level_energy_lookup = dict()
        self.element, datasections = self.read(datafile_name)
        self._model_data(datasections["model"])
        [l.set_index(i) for i, l in enumerate(self.levels)]
        self.index_dict = {(l.iso, l.i): l.index for l in self.levels}

        self.phxs = self._transition_data(datasections, "phxs")
        self.phot_ion = self._transition_data(datasections,"phot_ion")
        self.colex = self._transition_data(datasections, "colex2")
        self.colis = self._transition_data(datasections, "sampson")
        self.colon = self._transition_data(datasections, "colon2")
        self.ai = self._transition_data(datasections, "augxs")
        self.phis = self._transition_data(datasections, "phis")

    @staticmethod
    def read(datafile_name):
        with open(datafile_name) as f:
            datacontent = f.read()
        element = None
        match = re.search(r"([^\n]*\batom\b[^\n]*)", datacontent[:100])
        if match:
            element = match.group(1).split()[2]
        datacontent = re.sub(r"^c.*\n?", "", datacontent, flags=re.MULTILINE)
        datasections = re.split("^.*data|end data", datacontent, re.MULTILINE)
        datasections = [part.strip() for part in datasections if part.strip()]
        return element, {section.split()[1]: section for section in datasections}

    def _model_data(self, modelsection):
        enotsections = re.split("^.*enot|enot", modelsection)
        for enotsection in enotsections[1:]:
            self.parse_enot(enotsection)

    def parse_enot(self, enotsection):
        enotsection = [line.strip() for line in enotsection.split("\n") if line.strip()]
        enot = enotsection[0].split()
        iso_name = str(enot[1])
        iso_energy = float(enot[2])
        for level_line in enotsection[1:]:
            level = Level(level_line, iso_name, iso_energy)
            self.levels.append(level)
            self._statical_weight_lookup.update({(level.iso, level.i): level.g})
            self._iso_energy_lookup.update({(level.iso, level.i): level.iso_energy})
            self._level_energy_lookup.update({(level.iso, level.i): level.energy})

    def _transition_data(self, data, dataname):
        transition_list = list()
        try:
            datasection = data[dataname]
        except KeyError:
            return None
        for line in datasection.split("\n")[1:]:
            line = line.strip()
            if line.startswith("enot") or len(line) == 0 or line.startswith("c"):
                continue
            if dataname == "sampson":
                transition = Sampson(line)
            if dataname == "colex2":
                transition = Colex(line)
            if dataname == "augxs":
                transition = Augxs(line)
            if dataname == "colon2":
                transition = Colon(line)
            if dataname == "phot_ion":
                transition = PhotIon(line)
            if dataname == "phxs":
                transition = Phxs(line)
            if dataname == "phis":
                transition = Phis(line)

            transition.g1 = self.get_statistical_weight(transition.iso1, transition.i1)
            transition.g2 = self.get_statistical_weight(transition.iso2, transition.i2)
            transition.de = self.get_transition_energy(transition.iso1, transition.i1, transition.iso2, transition.i2)
            transition_list.append(transition)
        return transition_list

    def get_statistical_weight(self, iso, i):
        return self._statical_weight_lookup[(iso, i)]

    def get_transition_energy(self, iso1, i1, iso2, i2):
        iso_energy_l1 = self._iso_energy_lookup[(iso1, i1)]
        energy_l1 = self._level_energy_lookup[(iso1, i1)]
        iso_energy_l2 = self._iso_energy_lookup[(iso2, i2)]
        energy_l2 = self._level_energy_lookup[(iso2, i2)]
        return energy_l2 + iso_energy_l1 - energy_l1

    def get_level(self, iso, i):
        return next((l for l in self.levels if l.iso == iso and l.i == i), None)

    def get_levels(self, iso):
        return [l for l in self.levels if l.iso == iso]

    def get_index(self, iso, i):
        return self.get_level(iso, i).index

    def get_indices(self, iso):
        return [l.index for l in self.get_levels(iso)]

    def get_colis_index(self, iso1, i1):
        return [i for i, l in enumerate(self.colis) if l.iso1 == iso1 and l.i1 == i1]

    def get_colon(self, iso1, i1):
        return [l for l in self.colon if l.iso1 == iso1 and l.i1 == i1]

    def get_colon_iso(self, iso):
        return [l for l in self.colon if l.iso1 == iso]

    def get_key(self, index):
        return next((l.iso, l.i) for l in self.levels if l.index == index)

    def get_keys(self, indices):
        return [self.get_key(index) for index in indices]

    def get_phot_ion(self, iso1, i1, iso2=None, i2=None):
        if iso2 is None and i2 is None:
            return [t for t in self.phot_ion if t.iso1 == iso1 and t.i1 == i1]
        else:
            return next((t for t in self.phot_ion if t.iso1 == iso1 and t.i1 == i1 and t.iso2 == iso2 and t.i2 == i2), None)
        
    def get_phis(self, iso1, i1, iso2=None, i2=None):
        if iso2 is None and i2 is None:
            return [t for t in self.phis if t.iso1 == iso1 and t.i1 == i1]
        else:
            return next((t for t in self.phis if t.iso1 == iso1 and t.i1 == i1 and t.iso2 == iso2 and t.i2 == i2), None)
    def get_phxs(self, iso1, i1, iso2=None, i2=None):
        if iso2 is None and i2 is None:
            return [t for t in self.phxs if t.iso1 == iso1 and t.i1 == i1]
        else:
            return next((t for t in self.phxs if t.iso1 == iso1 and t.i1 == i1 and t.iso2 == iso2 and t.i2 == i2), None)
        
    def get_augx(self, iso1, i1, iso2=None, i2=None):
        if iso2 is None and i2 is None:
            return [t for t in self.ai if t.iso1 == iso1 and t.i1 == i1]
        else:
            return next((t for t in self.ai if t.iso1 == iso1 and t.i1 == i1 and t.iso2 == iso2 and t.i2 == i2), None)

    def remove_level_and_transitions(self, iso, i):
        self.remove_level(iso, i)
        self.remove_transitions(iso, i)

    def remove_level(self, iso, i):
        l = self.get_level(iso, i)
        self.levels.remove(l)

    def remove_transitions(self, iso, i):
        tbr = [t for t in self.phxs if (t.iso1 == iso and t.i1 == i) or (t.iso2 == iso and t.i2 == i)]
        [self.phxs.remove(t) for t in tbr]
        tbr = [t for t in self.phot_ion if (t.iso1 == iso and t.i1 == i) or (t.iso2 == iso and t.i2 == i)]
        [self.phot_ion.remove(t) for t in tbr]
        tbr = [t for t in self.colex if (t.iso1 == iso and t.i1 == i) or (t.iso2 == iso and t.i2 == i)]
        [self.colex.remove(t) for t in tbr]
        tbr = [t for t in self.colis if (t.iso1 == iso and t.i1 == i) or (t.iso2 == iso and t.i2 == i)]
        [self.colis.remove(t) for t in tbr]
        tbr = [t for t in self.ai if (t.iso1 == iso and t.i1 == i) or (t.iso2 == iso and t.i2 == i)]
        [self.ai.remove(t) for t in tbr]

    def resolve(self):
        isomax = self.find_isomax()
        for isolevels in [[l for l in self.levels if l.iso == iso] for iso in range(isomax + 1)]:
            for index, l in enumerate(isolevels):
                for t in [t for t in self.phxs if (t.iso1 == l.iso and t.i1 == l.i)]:
                    t.i = index + 1
                for t in [t for t in self.phxs if (t.iso2 == l.iso and t.i2 == l.i)]:
                    t.i = index + 1
                l.i = index + 1

    def write_level(self, f, l):
        config_string = "  ".join([str(c) for c in l.config])
        f.write(f"{'':4}elev{l.iso:>6}{l.i:>7}{'':>3}{l.name:<13}{l.g:>13g}.{l.energy:>13.3f}{'':>4}{config_string:<20}{l.nmax:<4}\n")

    def write_phxs(self, f):
        f.write(f"\ndata{'':4}phxs\n")
        for t in self.phxs:
            row = ("d", t.iso1, t.i1, t.iso2, t.i2, t.f, t.wavelength, t.width)
            fstring = "{:<3} {:<5} {:<6} {:<3} {:<6} {:<14.6E} {:<14.6E} {:<14.6E}\n".format(*row)
            f.write(fstring)
        f.write(f"\nend data\n")

    def write_enot(self, f, level):
        f.write(f"\n{'':<2}enot{level.iso:>5}{level.iso_name:>10}{level.iso_energy:10}\n")

    def find_isomax(self):
        return sorted([l.iso for l in self.levels])[-1]

    def write_header(self, f, nlmode):
        f.write(f"c atom {self.element}\n")
        level_description = "c n_shells\n"
        if nlmode:
            level_description = "c ls_shells\n"
        f.write(level_description)

    def write_model(self, f):
        iso_max = self.find_isomax()
        f.write(f"\ndata{'':4}model\n")
        for isolevels in [[l for l in self.levels if l.iso == iso] for iso in range(iso_max + 1)]:
            isolevels.sort(key=lambda x: x.i)
            self.write_enot(f, isolevels[0])
            [self.write_level(f, l) for l in isolevels]
        f.write(f"\nend data\n")

    def write(self, outfilename, nlmode):
        with open(f"{outfilename}", "w") as f:
            self.write_header(f, nlmode)
            self.write_model(f)
            self.write_phxs(f)
            f.close()
