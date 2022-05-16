use crate::types::*;
use std::mem::swap;
use std::time::*;
use std::cmp::max;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{ Hash, Hasher };

use std::sync::{ Arc, Mutex };
use crossbeam::thread;

#[derive(Debug, Hash)]
struct Bitmap
{
    data: Vec<u64>,
    size: usize,
    vec_size: usize,
}

impl Bitmap
{
    pub fn new(size: usize) -> Bitmap
    {
        let vec_size = (size - 1) / 64 + 1;
        let mut data = Vec::with_capacity(vec_size);
        data.resize(vec_size, 0u64);
        Bitmap {
            data, size, vec_size,
        }
    }

    pub fn clear(&mut self)
    {
        self.data.clear();
        self.data.resize(self.vec_size, 0);
    }

    pub fn get_bit(&self, index: usize) -> bool
    {
        self.data[index / 64] & (1 << (index & (64-1))) != 0
    }

    pub fn set_bit(&mut self, index: usize, value: bool)
    {
        let mask = 1 << (index & (64-1));
        if value
        {
            self.data[index / 64] |= mask;
        }
        else
        {
            self.data[index / 64] &= !mask;
        }
    }

}

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
enum GroupValue
{
    Nothing,
    High,
    Pullup,
    Pulldown,
    Vcc,
    Vss,
}


#[derive(Debug, Clone, PartialEq)]
struct HashResult
{
    next: Vec<Node>,
    inside: Vec<Node>,
    new_value: GroupValue,
}

#[derive(Debug, Clone)]
struct Connection
{
    t: Tran,
    other: Node,
}

impl Connection
{
    pub fn new(t: Tran, other: Node) -> Connection
    {
        Connection { t, other }
    }
}

#[derive(Debug)]
pub struct Group
{
    nodes: Vec<Node>,
    bitmap: Bitmap,
    value: GroupValue,

    cache: Vec<HashMap<u64, HashResult>>,
}

const CAPACITY: usize = 50;

impl Group
{
    pub fn new(nodes_count: usize) -> Group
    {
        Group { 
            nodes: Vec::with_capacity(CAPACITY),
            bitmap: Bitmap::new(nodes_count),
            value: GroupValue::Nothing,

            cache: vec![HashMap::new(); nodes_count],
        }
    }

    fn contains(&self, node: Node) -> bool
    {
        self.bitmap.get_bit(node)
    }

    #[inline]
    fn add(&mut self, node: Node)
    {
        self.nodes.push(node);
        self.bitmap.set_bit(node, true);
    }

    fn remove(&mut self, node: Node)
    {
        if let Some(i) = self.nodes.iter().position(|&x| x == node)
        {
            self.nodes.remove(i);
            self.bitmap.set_bit(node, false);
        }
    }

    #[inline]
    fn clear(&mut self, state: &State)
    {
        self.nodes.clear();
        self.value = GroupValue::Nothing;
        self.bitmap.clear();
    }

    #[inline]
    pub fn binary_value(&self) -> bool
    {
        match self.value
        {
            GroupValue::Vcc | GroupValue::Pullup | GroupValue::High => true,
            GroupValue::Vss | GroupValue::Pulldown | GroupValue::Nothing => false,
        }
    }

    #[inline]
    fn node_status(&self, state: &State, node: Node) -> (bool,bool,bool,bool,Vec<bool>)
    {
        (state.node_value(node),
         state.node_is_pulldown(node),
         state.node_is_pullup(node),
         self.contains(node),
         state.get_connections(node).iter().map(|c| state.tran_is_on(c.t)).collect())
    }

    // only simulates what happens if we add node. doesn't change state
    pub fn dry_run(&self, state: &State, node: Node, mut val: GroupValue)
        -> (bool, Option<Vec<Node>>, GroupValue)
    {
        if node == state.vss
        {
            val = GroupValue::Vss;
            return (false, None, val);
        }

        if node == state.vcc
        {
            if val != GroupValue::Vss
            {
                val = GroupValue::Vcc;
            }
            return (false, None, val);
        }

        if self.contains(node)
        {
            return (false, None, val);
        }

        // after this point the node will be added

        if state.node_is_pulldown(node)
        {
            val = max(GroupValue::Pulldown, val);
        }

        if state.node_is_pullup(node)
        {
            val = max(GroupValue::Pullup, val);
        }

        if val < GroupValue::High && state.node_value(node)
        {
            val = GroupValue::High;
        }

        let mut next = state.get_connections(node).iter()
            .filter(|c| state.tran_is_on(c.t))
            .map(|c| c.other).collect::<Vec<Node>>();

        next.sort();
        next.dedup();


        (true, Some(next), val)
    }

    pub fn add_node_to_group(&mut self, state: &State, node: Node)
    {
        //let n = Instant::now();

        // First: calculate the hash for the state of this node
        let mut h = DefaultHasher::new();

        self.value.hash(&mut h);
        self.node_status(state, node).hash(&mut h);        
        let hood = state.get_connections(node).iter();

        for s in hood.filter(|c| state.tran_is_on(c.t))
            .map(|c| self.node_status(state, c.other))
        {
            s.hash(&mut h);
        }

        let hash = h.finish();

        // Second: if this exact state has already been simulated,
        //         just use the previous result to advance 2 steps
        if self.cache[node].contains_key(&hash)
        {
            let result = &self.cache[node][&hash].clone();

            self.value = result.new_value;

            for &n in result.inside.iter()
            {
                self.add(n);
            }

            for &n in result.next.iter()
            {
                self.add_node_to_group(state, n);
            }

            return;
        }

        // Otherwise: simulate it and cache the result for the next time
        //let n = Instant::now();

        //// run the simulation twice, once on the current node,
        //// and once on all the nodes connected to it. this way we
        //// cache 2 steps of the simulation in one hash
        let (added, middle, mut new_val) = self.dry_run(state, node, self.value);

        let mut new: Vec<Node> = vec![];
        let mut next: Vec<Node> = vec![];

        if added
        {
            new.push(node); 
        }

        if let Some(half) = middle
        {
            if half.len() > 0
            {

            // run the simulation on all the connected nodes. combine
            // all the results from all of them. this reduces overall
            // redundancy.

            let next_gen = half.iter()
                .map(|&n| (n, self.dry_run(state, n, new_val)));

            let next_gen_val = next_gen.clone()
                .map(|r| r.1.2).max().unwrap_or(GroupValue::Nothing);
            
            let nodes = half.iter();

            // nodes that join the group
            let next_gen_inside = next_gen.clone()
                .filter(|r| r.1.0)
                .map(|r| r.0);

            // nodes that need to be simulated next
            let mut next_gen_next = next_gen
                .filter(|r| r.1.1 != None)
                .map(|r| r.1.1.unwrap()) // this really shouldn't crash
                .collect::<Vec<Vec<Node>>>()
                .concat();

            // get rid of repeats
            next_gen_next.sort();
            next_gen_next.dedup();
            if let Some(i) = next_gen_next.iter().position(|&x| x == node)
            {
                next_gen_next.remove(i);
            }

            new.extend(next_gen_inside);
            next.extend(next_gen_next);

            new_val = next_gen_val;

            }
        }

        // finally we have the result of the two step simulation
        // cache it in the hashmap to be used the next time
        let hash_result = HashResult {
                                inside: new,
                                next,
                                new_value: new_val,
                            };

        self.cache[node].insert(hash, hash_result);
        

        // finally call the function again. this time we have the
        // hash result and it'll be applied to the group
        self.add_node_to_group(state, node);
    }

    pub fn add_node_to_group_old(&mut self, state: &State, node: Node)
    {
            if node == state.vss
            {
                self.value = GroupValue::Vss;
                //continue;
                return;
            }
            
            if node == state.vcc
            {
                if self.value != GroupValue::Vss
                {
                    self.value = GroupValue::Vcc;
                }
                //continue;
                return;
            }
            
            if self.contains(node)
            {
                return;
            }
            
            self.add(node);

            if state.node_is_pulldown(node)
            {
                self.value = max(GroupValue::Pulldown, self.value);
            }
            else if state.node_is_pullup(node)
            {
                self.value = max(GroupValue::Pullup, self.value);
            }
            
            if self.value < GroupValue::High && state.node_value(node)
            {
                // node is not pull up but is high
                self.value = GroupValue::High;
            } 

            for c in state.get_connections(node).iter()
                .filter(|c| state.tran_is_on(c.t))
            {
                self.add_node_to_group_old(state, c.other);
            }
    }
}

pub struct State
{
    nodes_count: usize,
    trans_count: usize,
    
    vss: Node,
    vcc: Node,

    // - Node Data
    node_is_pullup: Bitmap,
    node_is_pulldown: Bitmap,
    node_value: Bitmap,

    // Gates that this node is connected to
    node_gates: Vec<Vec<Tran>>,
    //node_gates_count: [usize],

    // Nodes that depend on the value of this node
    node_depend: Vec<Vec<Node>>,
    node_depend_left: Vec<Vec<Node>>,
    // put this one | all in the same array like in the original
    // so it gets   V   cached more efficiently TODO
    node_connections: Vec<Connection>, 
    connections_offset: Vec<usize>,

    // - Transistor Data
    tran_gate: Vec<Node>,
    tran_a: Vec<Node>,
    tran_b: Vec<Node>,
    tran_is_on: Bitmap,


    // Nodes that we're working on
    current: Vec<Node>,

    // Nodes that we should work on
    queue: Vec<Node>,

}


impl State
{
    pub fn new(trans: &[Transistor], pullup: &[i32], vss: Node, vcc: Node)
        -> State
    {
        let nodes_count = pullup.len();
        let trans_count = trans.len();
        
        let mut s = State {
            nodes_count,
            trans_count,

            vss,
            vcc,

            node_is_pullup: Bitmap::new(nodes_count),
            node_is_pulldown: Bitmap::new(nodes_count),
            node_value: Bitmap::new(nodes_count),

            node_gates: vec![Vec::with_capacity(CAPACITY); nodes_count],

            node_depend: vec![Vec::with_capacity(CAPACITY); nodes_count],
            node_depend_left: vec![Vec::with_capacity(CAPACITY); nodes_count],

            node_connections: Vec::new(),
            connections_offset: vec![0usize; nodes_count+1],

            tran_gate: trans.into_iter().map(|t| t.0).collect(),
            tran_a: trans.into_iter().map(|t| t.1).collect(),
            tran_b: trans.into_iter().map(|t| t.2).collect(),
            tran_is_on: Bitmap::new(trans_count),

            current: Vec::new(),
            queue: Vec::new(),

        };

        pullup.iter().enumerate()
            .for_each(|(i,&x)| s.node_is_pullup.set_bit(i, x>0));

        let mut c_count = vec![0usize; nodes_count];

        for i in 0..trans_count
        {
            //let i = i as tran;
            let gate = s.tran_gate[i];
            let a = s.tran_a[i];
            let b = s.tran_b[i];

            s.node_gates[gate].push(i);
            c_count[a] += 1;
            c_count[b] += 1;
            //s.node_connections[a].push(Connection::new(i, b));
            //s.node_connections[b].push(Connection::new(i, a));
        }

        let total = c_count.iter().enumerate().fold(0, |off, (i,x)|
                            {
                                s.connections_offset[i] = off;
                                off + x
                            });
        s.connections_offset[nodes_count] = total;
        
        // reset the counters and now use as index
        c_count.iter_mut().for_each(|x| *x = 0);

        //println!("{}", total);
        s.node_connections.reserve(total);
        unsafe { s.node_connections.set_len(total); }

        // note: we can not zero it and just go backwards but it doesn't matter really
        for i in 0..trans_count
        {
            let a = s.tran_a[i];
            let b = s.tran_b[i];
            s.node_connections[s.connections_offset[a]
                + c_count[a]] = Connection::new(i, b);
            c_count[a] += 1;
            s.node_connections[s.connections_offset[b]
                + c_count[b]] = Connection::new(i, a);
            c_count[b] += 1;
        }

        for n in 0..nodes_count
        {
            for t in &s.node_gates[n]
            {
                let a = s.tran_a[*t];
                if a != s.vss && a != s.vcc
                {
                    s.node_depend[n].push(a);
                }
                
                let b = s.tran_b[*t];
                if b != s.vss && b != s.vcc
                {
                    s.node_depend[n].push(b);
                }

                let left = if a != s.vss && a != s.vcc { a } else { b };
                s.node_depend_left[n].push(left); 
            }
        }

        s
    }


    pub fn node_is_pulldown(&self, node: Node) -> bool
    {
        self.node_is_pulldown.get_bit(node)
    }

    pub fn node_is_pullup(&self, node: Node) -> bool
    {
        self.node_is_pullup.get_bit(node)
    }

    pub fn node_value(&self, node: Node) -> bool
    {
        self.node_value.get_bit(node)
    }

    pub fn tran_is_on(&self, node: Node) -> bool
    {
        self.tran_is_on.get_bit(node)
    }
    
    #[inline]
    fn get_connections(&self, node: Node) -> &[Connection]
    {
        &self.node_connections[
            self.connections_offset[node]
          ..self.connections_offset[node+1]]
    }

    #[inline]
    fn queue_clear(&mut self)
    {
        self.queue.clear();
    }

    #[inline]
    pub fn list_switch(&mut self)
    {
        swap(&mut self.current, &mut self.queue);
    }

    #[inline]
    pub fn recalc_node(&mut self, node: Node, mut group: &mut Group)
    {
        group.clear(&self);
        group.add_node_to_group(self, node);

            let newv = group.binary_value();
        // set all nodes to the group state
        // check and switch all the transistors connected
        // collect all nodes connected to those transistors
        for n in group.nodes.iter()
        {
             if self.node_value(*n) != newv
             {
                self.node_value.set_bit(*n, newv);
                for t in &self.node_gates[*n]
                {
                    self.tran_is_on.set_bit(*t, newv);
                }

                let depend = if newv 
                    { &self.node_depend_left } else { &self.node_depend };
                
                for nn in depend[*n].iter()
                {
                    self.queue.push(*nn);
                }
             }
        }
    }

    #[inline]
    pub fn recalc_node_list(&mut self, group: &mut Group)
    {
        //let mut tries = 1;
        //let mut total_time = 0;
        for _ in 0..100
        {
            let n = Instant::now();
            self.list_switch();

            if self.current.len() == 0
            {
                break;
            }

            self.queue_clear();


            for n in self.current.clone()
            {
                self.recalc_node(n, group);
            }
        }

        self.queue_clear();
    }

    #[inline]
    pub fn stabilize_chip(&mut self, group: &mut Group)
    {
        for n in 0..self.nodes_count
        {
            self.queue.push(n);
        }
        self.recalc_node_list(group);
    }

    #[inline]
    pub fn set_node(&mut self, node: Node, v: bool, group: &mut Group)
    {
        self.node_is_pullup.set_bit(node, v);
        self.node_is_pulldown.set_bit(node, !v);
        self.queue.push(node);

        self.recalc_node_list(group);
    }

    /*
    #[inline]
    pub fn set_node_no_recalc(&mut self, node: Node, v: bool)
    {
        self.node_is_pullup[node] = v;
        self.node_is_pulldown[node] = !v;
        self.queue.push(node);
    }
    */
    #[inline]
    pub fn read_nodes(&self, list: &[Node]) -> u16
    {
        let mut result = 0;
        for n in list
        {
            result <<= 1;
            result |= self.node_value(*n) as u16;
        }
        result
    }
    
    #[inline]
    pub fn read_node(&self, node: Node) -> bool
    {
        self.node_value(node)
    }
    

    #[inline]
    pub fn write_nodes(&mut self, list: &[Node], mut value: u16, group: &mut Group)
    {
        for n in list.iter().rev()
        {
            self.set_node(*n, (value & 1) != 0, group);
            value >>= 1;
        }
        self.recalc_node_list(group);
    }

}
