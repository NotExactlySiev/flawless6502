use std::iter::Map;

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug, Clone, Copy, Hash)]
pub enum GroupValue
{
    Nothing,
    High,
    Pullup,
    Pulldown,
    Vcc,
    Vss,
}

pub type Node = usize;
pub type Tran = usize;
pub type NodeState = 
    (bool, bool, bool, bool, Vec<bool>);

pub type NodeFullState =
    (GroupValue, NodeState, Vec<NodeState>);

pub struct Transistor(pub Node, pub Node, pub Node);

