import collections
from itertools import product, combinations
import numpy as np
import numba
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, cdist, squareform
import ase.io.cif
import ase.data
from ase.spacegroup.spacegroup import parse_sitesym


def _extract_motif_and_cell(periodic_set):

    asymmetric_unit, multiplicities = None, None

    if isinstance(periodic_set, PeriodicSet):
        motif, cell = periodic_set.motif, periodic_set.cell

        if 'asymmetric_unit' in periodic_set.tags and 'wyckoff_multiplicities' in periodic_set.tags:
            asymmetric_unit = periodic_set.asymmetric_unit
            multiplicities = periodic_set.wyckoff_multiplicities

    elif isinstance(periodic_set, np.ndarray):
        motif, cell = periodic_set, None
    else:
        motif, cell = periodic_set[0], periodic_set[1]

    return motif, cell, asymmetric_unit, multiplicities


def _collapse_into_groups(overlapping):

    overlapping = squareform(overlapping)
    group_nums = {} 
    group = 0
    for i, row in enumerate(overlapping):
        if i not in group_nums:
            group_nums[i] = group
            group += 1

            for j in np.argwhere(row).T[0]:
                if j not in group_nums:
                    group_nums[j] = group_nums[i]

    groups = collections.defaultdict(list)
    for row_ind, group_num in sorted(group_nums.items()):
        groups[group_num].append(row_ind)
    groups = list(groups.values())

    return groups


@numba.njit()
def _dist(xy, z):
    s = z ** 2
    for val in xy:
        s += val ** 2
    return s

@numba.njit()
def _distkey(pt):
    s = 0
    for val in pt:
        s += val ** 2
    return s

def _generate_integer_lattice(dims):

    ymax = collections.defaultdict(int)
    d = 0

    if dims == 1:
        yield np.array([[0]])
        while True:
            d += 1
            yield np.array([[-d], [d]])

    while True:
        positive_int_lattice = []
        while True:
            batch = []
            for xy in product(range(d+1), repeat=dims-1):
                if _dist(xy, ymax[xy]) <= d**2:
                    batch.append((*xy, ymax[xy]))
                    ymax[xy] += 1
            if not batch:
                break
            positive_int_lattice += batch
        positive_int_lattice.sort(key=_distkey)
        
        int_lattice = []
        for p in positive_int_lattice:
            int_lattice.append(p)
            for n_reflections in range(1, dims+1):
                for indexes in combinations(range(dims), n_reflections):
                    if all((p[i] for i in indexes)):
                        p_ = list(p)
                        for i in indexes:
                            p_[i] *= -1
                        int_lattice.append(p_)

        yield np.array(int_lattice)
        d += 1


def _generate_concentric_cloud(motif, cell):

    int_lattice_generator = _generate_integer_lattice(cell.shape[0])

    while True:
        int_lattice = next(int_lattice_generator) @ cell
        yield np.concatenate([motif + translation for translation in int_lattice])


def _nearest_neighbours(motif, cell, k, asymmetric_unit=None):

    if asymmetric_unit is not None:
        asym_unit = motif[asymmetric_unit]
    else:
        asym_unit = motif

    cloud_generator = _generate_concentric_cloud(motif, cell)
    n_points = 0
    cloud = []
    while n_points <= k:
        l = next(cloud_generator)
        n_points += l.shape[0]
        cloud.append(l)
    cloud.append(next(cloud_generator))
    cloud = np.concatenate(cloud)

    tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
    pdd_, inds = tree.query(asym_unit, k=k+1, workers=-1)
    pdd = np.zeros_like(pdd_)

    while not np.allclose(pdd, pdd_, atol=1e-12, rtol=0):
        pdd = pdd_
        cloud = np.vstack((cloud,
                           next(cloud_generator),
                           next(cloud_generator)))
        tree = KDTree(cloud, compact_nodes=False, balanced_tree=False)
        pdd_, inds = tree.query(asym_unit, k=k+1, workers=-1)

    return pdd_[:, 1:], cloud, inds[:, 1:]


@numba.njit(cache=True)
def _network_simplex(source_demands, sink_demands, network_costs):

    n_sources, n_sinks = source_demands.shape[0], sink_demands.shape[0]
    network_costs = network_costs.ravel()
    fp_multiplier = np.array([1000000], dtype=np.int64)
    nodes = np.arange(n_sources + n_sinks).astype(np.int64)
    source_d_fp = source_demands * fp_multiplier.astype(np.int64)
    source_d_int = source_d_fp.astype(np.int64)
    sink_d_fp = sink_demands * fp_multiplier.astype(np.int64)
    sink_d_int = sink_d_fp.astype(np.int64)
    source_sum = np.sum(source_d_int)
    sink_sum = np.sum(sink_d_int)

    if  source_sum < sink_sum:
        source_ind = np.argmax(source_d_int)
        source_d_int[source_ind] += sink_sum - source_sum
    elif sink_sum < source_sum:
        sink_ind = np.argmax(sink_d_int)
        sink_d_int[sink_ind] += source_sum - sink_sum

    demands = np.concatenate((-source_d_int, sink_d_int)).astype(np.int64)
    conn_tails = np.array([i for i in range(n_sources) for _ in range(n_sinks)],
                          dtype=np.int64)
    conn_heads = np.array([j + n_sources for _ in range(n_sources) for j in range(n_sinks)],
                          dtype=np.int64)

    dummy_tails = []
    dummy_heads = []

    for node, demand in np.ndenumerate(demands):
        if demand > 0:
            dummy_tails.append(node[0])
            dummy_heads.append(-1)
        else:
            dummy_tails.append(-1)
            dummy_heads.append(node[0])

    tails = np.concatenate((conn_tails, np.array(dummy_heads).T)).astype(np.int64)
    heads = np.concatenate((conn_heads, np.array(dummy_heads).T)).astype(np.int64)
    network_costs = network_costs * fp_multiplier
    network_capac = np.array([np.array([source_demands[i], sink_demands[j]]).min()
                              for i in range(n_sources)
                              for j in range(n_sinks)],
                             dtype=np.float64) * fp_multiplier
    faux_inf = 3 * np.max(np.array((np.sum(network_capac.astype(np.int64)),
                                    np.sum(np.absolute(network_costs)),
                                    np.max(np.absolute(demands))),
                                   dtype=np.int64))
    costs = np.concatenate((network_costs, np.ones(nodes.shape[0]) * faux_inf)).astype(np.int64)
    capac = np.concatenate((network_capac, np.ones(nodes.shape[0]) * fp_multiplier)).astype(np.int64)
    e = conn_tails.shape[0]
    n = nodes.shape[0]
    flows = np.concatenate((np.zeros(e), np.array([abs(d) for d in demands]))).astype(np.int64)
    potentials = np.array([faux_inf if d <= 0 else -faux_inf for d in demands]).T
    parent = np.concatenate((np.ones(n) * -1, np.array([-2]))).astype(np.int64)
    edge = np.arange(e, e + n).astype(np.int64)
    size = np.concatenate((np.ones(n), np.array([n + 1]))).astype(np.int64)
    next_node = np.concatenate((np.arange(1, n), np.array([-1, 0]))).astype(np.int64)
    prev_node = np.arange(-1, n)
    last_node = np.concatenate((np.arange(n), np.array([n - 1]))).astype(np.int64)
    f = 0

    while True:
        i, p, q, f = _find_entering_edges(e, f, tails, heads, costs, potentials, flows)
        if p == -1:
            break

        cycle_nodes, cycle_edges = _find_cycle(i, p, q, size, edge, parent)
        j, s, t = _find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads)
        _augment_flow(cycle_nodes, cycle_edges, _residual_capacity(j, s, capac, flows, tails), tails, flows)

        if i != j:
            if parent[t] != s:
                s, t = t, s
            if np.where(cycle_edges == i)[0][0] > np.where(cycle_edges == j)[0][0]:
                p, q = q, p

            _remove_edge(s, t, size, prev_node, last_node, next_node, parent, edge)
            _make_root(q, parent, size, last_node, prev_node, next_node, edge)
            _add_edge(i, p, q, next_node, prev_node, last_node, size, parent, edge)
            _update_potentials(i, p, q, heads, potentials, costs, last_node, next_node)

    flow_cost = 0
    final_flows = flows[:e].astype(np.float64)
    edge_costs = costs[:e].astype(np.float64)

    for arc_ind, flow in np.ndenumerate(final_flows):
        flow_cost += flow * edge_costs[arc_ind]

    final = flow_cost / fp_multiplier
    final = final.astype(np.float64)
    final = final / fp_multiplier
    final_flows = final_flows / fp_multiplier

    return final[0], final_flows

@numba.njit(cache=True)
def _reduced_cost(i, costs, potentials, tails, heads, flows):
    c = costs[i] - potentials[tails[i]] + potentials[heads[i]]
    if flows[i] == 0:
        return c
    else:
        return -c

@numba.njit(cache=True)
def _find_entering_edges(e, f, tails, heads, costs, potentials, flows):

    B = np.int64(np.ceil(np.sqrt(e)))
    M = (e + B - 1) // B
    m = 0

    while m < M:
        l = f + B
        if l <= e:
            edge_inds = np.arange(f, l)
        else:
            l -= e
            edge_inds = np.concatenate((np.arange(f, e), np.arange(l)))

        f = l
        r_costs = np.empty(edge_inds.shape[0])

        for y, z in np.ndenumerate(edge_inds):
            r_costs[y] = _reduced_cost(z, costs, potentials, tails, heads, flows)

        h = np.argmin(r_costs)
        i = edge_inds[h]
        c = _reduced_cost(i, costs, potentials, tails, heads, flows)
        p = q = -1

        if c >= 0:
            m += 1

        else:
            if flows[i] == 0:
                p = tails[i]
                q = heads[i]
            else:
                p = heads[i]
                q = tails[i]
            return i, p, q, f
        
    return -1, -1, -1, -1


@numba.njit(cache=True)
def _find_apex(p, q, size, parent):
    size_p = size[p]
    size_q = size[q]

    while True:
        while size_p < size_q:
            p = parent[p]
            size_p = size[p]
        while size_p > size_q:
            q = parent[q]
            size_q = size[q]
        if size_p == size_q:
            if p != q:
                p = parent[p]
                size_p = size[p]
                q = parent[q]
                size_q = size[q]
            else:
                return p

@numba.njit(cache=True)
def _trace_path(p, w, edge, parent):
    cycle_nodes = [p]
    cycle_edges = []

    while p != w:
        cycle_edges.append(edge[p])
        p = parent[p]
        cycle_nodes.append(p)

    return cycle_nodes, cycle_edges

@numba.njit(cache=True)
def _find_cycle(i, p, q, size, edge, parent):
    w = _find_apex(p, q, size, parent)
    cycle_nodes, cycle_edges = _trace_path(p, w, edge, parent)
    cycle_nodes = np.array(cycle_nodes[::-1])
    cycle_edges = np.array(cycle_edges[::-1])

    if cycle_edges.shape[0] < 1:
        cycle_edges = np.concatenate((cycle_edges, np.array([i])))
    elif cycle_edges[0] != i:
        cycle_edges = np.concatenate((cycle_edges, np.array([i])))

    cycle_nodes_rev, cycle_edges_rev = _trace_path(q, w, edge, parent)
    cycle_nodes = np.concatenate((cycle_nodes, np.int64(cycle_nodes_rev[:-1])))
    cycle_edges = np.concatenate((cycle_edges, np.int64(cycle_edges_rev)))
    return cycle_nodes, cycle_edges

@numba.njit(cache=True)
def _residual_capacity(i, p, capac, flows, tails):
    if tails[np.int64(i)] == np.int64(p):
        return capac[np.int64(i)] - flows[np.int64(i)]
    else:
        return flows[np.int64(i)]

@numba.njit(cache=True)
def _find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads):
    cyc_edg_rev = np.flip(cycle_edges)
    cyc_nod_rev = np.flip(cycle_nodes)
    res_caps = []
    i = 0
    for edg in cyc_edg_rev:
        res_caps.append(_residual_capacity(edg, cyc_nod_rev[i], capac, flows, tails))
        i += 1

    res_caps = np.array(res_caps)
    j = cyc_edg_rev[np.argmin(res_caps)]
    s = cyc_nod_rev[np.argmin(res_caps)]
    t = heads[np.int64(j)] if tails[np.int64(j)] == s else tails[np.int64(j)]
    return j, s, t

@numba.njit(cache=True)
def _augment_flow(cycle_nodes, cycle_edges, f, tails, flows):
    for i, p in zip(cycle_edges, cycle_nodes):
        if tails[np.int64(i)] == np.int64(p):
            flows[np.int64(i)] += f
        else:
            flows[np.int64(i)] -= f

@numba.njit(cache=True)
def _trace_subtree(p, last_node, next_node):
    tree = []
    tree.append(p)
    l = last_node[p]
    while p != l:
        p = next_node[p]
        tree.append(p)
    return np.array(tree, dtype=np.int64)

@numba.njit(cache=True)
def _remove_edge(s, t, size, prev, last, next_node, parent, edge):
    size_t = size[t]
    prev_t = prev[t]
    last_t = last[t]
    next_last_t = next_node[last_t]
    parent[t] = -2
    edge[t] = -2
    next_node[prev_t] = next_last_t
    prev[next_last_t] = prev_t
    next_node[last_t] = t
    prev[t] = last_t

    while s != np.int64(-2):
        size[s] -= size_t
        if last[s] == last_t:
            last[s] = prev_t
        s = parent[s]

@numba.njit(cache=True)
def _make_root(q, parent, size, last, prev, next_node, edge):
    ancestors = []
    while q != np.int64(-2):
        ancestors.append(q)
        q = parent[q]
    ancestors.reverse()
    ancestors_min_last = ancestors[:-1]
    next_ancs = ancestors[1:]

    for p, q in zip(ancestors_min_last, next_ancs):
        size_p = size[p]
        last_p = last[p]
        prev_q = prev[q]
        last_q = last[q]
        next_last_q = next_node[last_q]
        parent[p] = q
        parent[q] = -2
        edge[p] = edge[q]
        edge[q] = -2
        size[p] = size_p - size[q]
        size[q] = size_p
        next_node[prev_q] = next_last_q
        prev[next_last_q] = prev_q
        next_node[last_q] = q
        prev[q] = last_q

        if last_p == last_q:
            last[p] = prev_q
            last_p = prev_q

        prev[p] = last_q
        next_node[last_q] = p
        next_node[last_p] = q
        prev[q] = last_p
        last[q] = last_p

@numba.njit(cache=True)
def _add_edge(i, p, q, next_node, prev_node, last, size, parent, edge):
    last_p = last[p]
    next_last_p = next_node[last_p]
    size_q = size[q]
    last_q = last[q]
    parent[q] = p
    edge[q] = i
    next_node[last_p] = q
    prev_node[q] = last_p
    prev_node[next_last_p] = last_q
    next_node[last_q] = next_last_p

    while p != np.int64(-2):
        size[p] += size_q
        if last[p] == last_p:
            last[p] = last_q
        p = parent[p]

@numba.njit(cache=True)
def _update_potentials(i, p, q, heads, potentials, costs, last_node, next_node):
    if q == heads[i]:
        d = potentials[p] - costs[i] - potentials[q]
    else:
        d = potentials[p] + costs[i] - potentials[q]

    tree = _trace_subtree(q, last_node, next_node)
    for q in tree:
        potentials[q] += d


##### PeriodicSet class represents a crystal as a periodic set #####

class PeriodicSet:
    
    def __init__(self, motif, cell, name=None, **kwargs):

        self.motif = motif
        self.cell = cell
        self.name = name
        self.tags = kwargs

    def __getattr__(self, attr):

        if 'tags' not in self.__dict__:
            self.tags = {}
        if attr in self.tags:
            return self.tags[attr]

        raise AttributeError(f"{self.__class__.__name__} object has no attribute or tag {attr}")


##### Reading .cifs #####

def _expand_asym_unit(asym_unit, sitesym):

    rotations, translations = parse_sitesym(sitesym)
    all_sites = []
    asym_inds = [0]
    multiplicities = []
    inverses = []

    for inv, site in enumerate(asym_unit):
        multiplicity = 0

        for rot, trans in zip(rotations, translations):
            site_ = np.mod(np.dot(rot, site) + trans, 1)

            if not all_sites:
                all_sites.append(site_)
                inverses.append(inv)
                multiplicity += 1
                continue

            diffs1 = np.abs(site_ - all_sites)
            diffs2 = np.abs(diffs1 - 1)
            mask = np.all((diffs1 <= 1e-3) | (diffs2 <= 1e-3), axis=-1)

            if not np.any(mask):
                all_sites.append(site_)
                inverses.append(inv)
                multiplicity += 1

        if multiplicity > 0:
            multiplicities.append(multiplicity)
            asym_inds.append(len(all_sites))

    frac_motif = np.array(all_sites)
    asym_inds = np.array(asym_inds[:-1])
    multiplicities = np.array(multiplicities)
    return frac_motif, asym_inds, multiplicities, inverses


def read_cif(path):
    
    crystals = []
    for block in ase.io.cif.parse_cif(path):
        
        cell = block.get_cell().array
        asym_unit = [block.get(name) for name in ['_atom_site_fract_x',
                                                  '_atom_site_fract_y',
                                                  '_atom_site_fract_z',]]
        asym_unit = np.mod(np.array(asym_unit).T, 1)
        asym_types = [ase.data.atomic_numbers[s] for s in block.get_symbols()]
        try:
            sitesym = block['_symmetry_equiv_pos_as_xyz']
        except:
            sitesym = block['_space_group_symop_operation_xyz']
        if isinstance(sitesym, str):
            sitesym = [sitesym]

        frac_motif, asym_inds, multiplicities, inverses = _expand_asym_unit(asym_unit, sitesym)
        full_types = np.array([asym_types[i] for i in inverses])
        motif = frac_motif @ cell

        tags = {
            'name': block.name,
            'asymmetric_unit': asym_inds,
            'wyckoff_multiplicities': multiplicities,
            'types': full_types,
            'cellpar': block.get_cellpar(),
        }
        
        crystals.append(PeriodicSet(motif, cell, **tags))
    
    if len(crystals) == 1:
        return crystals[0]
    else:
        return crystals


def pdd_finite(motif, lexsort=True, collapse=True, collapse_tol=1e-4):

    dm = squareform(pdist(motif))
    m = motif.shape[0]
    dists = np.sort(dm, axis=-1)[:, 1:]
    weights = np.full((m, ), 1 / m)
    groups = [[i] for i in range(len(dists))]

    if collapse:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping < collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            ordering = [group[0] for group in groups]
            dists = dists[ordering]

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        groups = [groups[i] for i in lex_ordering]
        pdd = pdd[lex_ordering]

    return pdd


def pdd(periodic_set, k, lexsort=True, collapse=True, collapse_tol=1e-4):

    motif, cell, asymmetric_unit, multiplicities = _extract_motif_and_cell(periodic_set)
    dists, _, _ = _nearest_neighbours(motif, cell, k, asymmetric_unit=asymmetric_unit)
    groups = [[i] for i in range(len(dists))]

    if multiplicities is None:
        weights = np.full((motif.shape[0], ), 1 / motif.shape[0])
    else:
        weights = multiplicities / np.sum(multiplicities)

    if collapse:
        overlapping = pdist(dists, metric='chebyshev')
        overlapping = overlapping < collapse_tol
        if overlapping.any():
            groups = _collapse_into_groups(overlapping)
            weights = np.array([sum(weights[group]) for group in groups])
            ordering = [group[0] for group in groups]
            dists = dists[ordering]

    pdd = np.hstack((weights[:, None], dists))

    if lexsort:
        lex_ordering = np.lexsort(np.rot90(dists))
        pdd = pdd[lex_ordering]

    return pdd


def emd(pdd, pdd_, metric='chebyshev', **kwargs):

    dm = cdist(pdd[:, 1:], pdd_[:, 1:], metric=metric, **kwargs)
    emd_dist, _ = _network_simplex(pdd[:, 0], pdd_[:, 0], dm)

    return emd_dist
