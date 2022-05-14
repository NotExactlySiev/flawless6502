use crate::types::*;
use std::mem::swap;
use std::time::*;
use std::cmp::max;
use std::collections::HashMap;

use std::sync::{ Arc, Mutex };
use crossbeam::thread;

#[derive(Debug)]
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
        let data = Vec::with_capacity(vec_size);
        data.resize(vec_size, 0u64);
        Bitmap {
            data, size, vec_size,
        }
    }

    pub fn clear(&mut self)
    {
        data.clear();
        data.resize(vec_size, 0);
    }

    pub fn get_bit(&self, index: usize) -> bool
    {
        data[index / 64] & (1 << (index & (64-1)))
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

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy)]
enum GroupValue
{
    Nothing,
    High,
    Pullup,
    Pulldown,
    Vcc,
    Vss,
}


#[derive(Debug, Clone)]
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
    group: Vec<Node>,
    bitmap: Bitmap,
    group_has_value: GroupValue,

    cache: Vec<HashMap<u64, HashResult>>,
}

const CAPACITY: usize = 50;

impl Group
{
    pub fn new(nodes_count: usize) -> Group
    {
        Group { 
            group: Vec::with_capacity(CAPACITY),
            bitmap: Bitmap::new(nodes_count),
            group_has_value: GroupValue::Nothing,

            cache: Vec::with_capacity(nodes_count),
        }
    }

    #[inline]
    fn group_add(&mut self, node: Node)
    {
        self.group.push(node);
        self.group_contains[node] = true;
    }

    #[inline]
    fn group_clear(&mut self, state: &State)
    {
        //let n = Instant::now();
        self.group.clear();
        self.group_has_value = GroupValue::Nothing;
        // if we use memset here it'll be faster i think
        self.group_contains.clear();
        self.group_contains.resize(state.nodes_count, false);
    }

    #[inline]
    pub fn group_value(&self) -> bool
    {
        match self.group_has_value
        {
            GroupValue::Vcc | GroupValue::Pullup | GroupValue::High => true,
            GroupValue::Vss | GroupValue::Pulldown | GroupValue::Nothing => false,
        }
    }


    pub fn add_node_to_group(&mut self, state: &State, node: Node)
    {
        // somehow the non-recursive version is slower lol
        //const CAP: usize = 5;
        //let mut stack = Vec::with_capacity(CAP);
        //stack.push(node);

        //while let Some(node) = stack.pop()
        //{
            if node == state.vss
            {
                self.group_has_value = GroupValue::Vss;
                //continue;
                return;
            }
            
            if node == state.vcc
            {
                if self.group_has_value != GroupValue::Vss
                {
                    self.group_has_value = GroupValue::Vcc;
                }
                //continue;
                return;
            }
            
           
            if self.group_contains(node)
            {
                //println!("why are we here");
                //continue;
                return;
            }
            
            self.group_add(node);

            if state.node_is_pulldown(node)
            {
                self.group_has_value = max(GroupValue::Pulldown, self.group_has_value);
            }
            else if state.node_is_pullup(node)
            {
                self.group_has_value = max(GroupValue::Pullup, self.group_has_value);
            }
            
            if self.group_has_value < GroupValue::High && state.node_value[node]
            {
                // node is not pull up but is high
                self.group_has_value = GroupValue::High;
            } 

            /*if state.node_connections[node].iter()
                .filter(|c| state.tran_is_on[c.t])
                .count() < 3
            {
                
            }*/
            

            for c in state.get_connections(node).iter()
                        .filter(|c| state.tran_is_on[c.t])
            {
                //if !self.group_contains[c.other]
                //{
                    self.add_node_to_group(state, c.other);
                //}
            }
            /*
            state.node_connections[node].iter()
                .filter(|c| state.tran_is_on[c.t])
                .for_each(|c| self.add_node_to_group(&state, c.other));
            */
        
      //  }
    
    }
}

pub struct State
{
    nodes_count: usize,
    trans_count: usize,
    
    vss: Node,
    vcc: Node,

    // - Node Data
    node_is_pullup: Vec<bool>,
    node_is_pulldown: Vec<bool>,
    node_value: Vec<bool>,

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
    tran_is_on: Vec<bool>,


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

            node_is_pullup:
                pullup.into_iter()
                .map(|x| *x > 0)
                .collect::<Vec<bool>>(),
        
            node_is_pulldown: vec![false; nodes_count],  

            node_value: vec![false; nodes_count],

            node_gates: vec![Vec::with_capacity(CAPACITY); nodes_count],

            node_depend: vec![Vec::with_capacity(CAPACITY); nodes_count],
            node_depend_left: vec![Vec::with_capacity(CAPACITY); nodes_count],

            node_connections: Vec::new(),
            connections_offset: vec![0usize; nodes_count+1],

            tran_gate: trans.into_iter().map(|t| t.0).collect(),
            tran_a: trans.into_iter().map(|t| t.1).collect(),
            tran_b: trans.into_iter().map(|t| t.2).collect(),
            tran_is_on: vec![false; trans_count],

            current: Vec::new(),
            queue: Vec::new(),

        };

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
        group.group_clear(&self);

        group.add_node_to_group(self, node);
        
        let newv = group.group_value();
        // set all nodes to the group state
        // check and switch all the transistors connected
        // collect all nodes connected to those transistors
        for n in &group.group
        {
             if self.node_value[*n] != newv
             {
                self.node_value[*n] = newv;
                for t in &self.node_gates[*n]
                {
                    self.tran_is_on[*t] = newv;
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
        self.node_is_pullup[node] = v;
        self.node_is_pulldown[node] = !v;
        self.queue.push(node);

        self.recalc_node_list(group);
    }

    #[inline]
    pub fn set_node_no_recalc(&mut self, node: Node, v: bool)
    {
        self.node_is_pullup[node] = v;
        self.node_is_pulldown[node] = !v;
        self.queue.push(node);
    }

    #[inline]
    pub fn read_nodes(&self, list: &[Node]) -> u16
    {
        let mut result = 0;
        for n in list
        {
            result <<= 1;
            result |= self.node_value[*n] as u16;
        }
        result
    }
    
    #[inline]
    pub fn read_node(&self, node: Node) -> bool
    {
        self.node_value[node]
    }
    

    #[inline]
    pub fn write_nodes(&mut self, list: &[Node], mut value: u16, group: &mut Group)
    {
        for n in list.iter().rev()
        {
            self.set_node_no_recalc(*n, (value & 1) != 0);
            value >>= 1;
        }
        self.recalc_node_list(group);
    }

}
