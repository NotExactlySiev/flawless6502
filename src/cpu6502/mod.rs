pub mod data;

use data::*;
use crate::sim::{ State, Group };
use std::thread;

pub struct CPU6502
{
    pub state: State,
    group: Group,
    group2: Group,
}

impl CPU6502
{
    // TODO: expose pins to the chip here
    pub fn new() -> CPU6502
    {
        CPU6502 {
            state: State::new(&transdefs, &node_is_pullup, vss, vcc),
            group: Group::new(node_is_pullup.len(), transdefs.len()),
            group2: Group::new(node_is_pullup.len(), transdefs.len()),
        }
    }
    
    pub fn initialize(&mut self)
    {
        self.state.set_node(res, false, &mut self.group);
        self.state.set_node(clk0, true, &mut self.group);
        self.state.set_node(rdy, true, &mut self.group);
        self.state.set_node(so, false, &mut self.group);
        self.state.set_node(irq, true, &mut self.group);
        self.state.set_node(nmi, true, &mut self.group);
        
        self.state.stabilize_chip(&mut self.group);
        
        // hold reset for 8 cycles and release
        for i in 0..16
        {
            self.cpu_step();
        }

        self.state.set_node(res, true, &mut self.group);
    }

    pub fn cpu_step(&mut self)
    {
        // TODO: it seems slow to recalc every time we change a signle node
        // maybe have a method like write_not_recalc
        let clk = self.state.read_node(clk0);
        self.state.set_node(clk0, !clk, &mut self.group);
        self.state.recalc_node_list(&mut self.group);
    }

    pub fn read_clk(&self) -> u8
    {
        self.state.read_nodes(&[clk0]) as u8
    }

    pub fn read_A(&self) -> u8
    {
        self.state.read_nodes(&[a7,a6,a5,a4,a3,a2,a1,a0]) as u8
    }

    pub fn read_X(&self) -> u8
    {
        self.state.read_nodes(&[x7,x6,x5,x4,x3,x2,x1,x0]) as u8
    }

    pub fn read_Y(&self) -> u8
    {
        self.state.read_nodes(&[y7,y6,y5,y4,y3,y2,y1,y0]) as u8
    }

    pub fn read_S(&self) -> u8
    {
        self.state.read_nodes(&[s7,s6,s5,s4,s3,s2,s1,s0]) as u8
    }

    pub fn read_P(&self) -> u8
    {
        self.state.read_nodes(&[p7,p6,p5,p4,p3,p2,p1,p0]) as u8
    }
    
    pub fn read_addr(&self) -> u16
    {
        self.state.read_nodes(&[ab15, ab14, ab13, ab12, ab11, ab10, ab9, ab8,
                       ab7, ab6, ab5, ab4, ab3, ab2, ab1, ab0])
    }

    pub fn read_pc(&self) -> u16
    {

        self.state.read_nodes(&[pch7, pch6, pch5, pch4, pch3, pch2, pch1, pch0,
            pcl7, pcl6, pcl5, pcl4, pcl3, pcl2, pcl1, pch0]) as u16
    }

    pub fn write_pc(&mut self, value: u16)
    {
        self.state.write_nodes(&[pch7, pch6, pch5, pch4, pch3, pch2, pch1, pch0, pcl7, pcl6, pcl5, pcl4, pcl3, pcl2, pcl1, pch0], value, &mut self.group)
    }


    pub fn write_addr(&mut self, value: u16)
    {
        self.state.write_nodes(&[ab15, ab14, ab13, ab12, ab11, ab10, ab9, ab8,
                       ab7, ab6, ab5, ab4, ab3, ab2, ab1, ab0], value, &mut self.group)
    }

    pub fn write_data(&mut self, value: u8)
    {
        self.state.write_nodes(&[db7, db6, db5, db4, db3, db2, db1, db0], value as u16, &mut self.group);
    }
    
    pub fn read_rw(&self) -> bool
    {
        self.state.read_node(rw)
    }

    pub fn read_data(&self) -> u8
    {
        self.state.read_nodes(&[db7, db6, db5, db4, db3, db2, db1, db0]) as u8
    }

    pub fn parallel_test(&mut self)
    {
        //self.state.recalc_node_dry()
        /*
        thread::spawn(||
        {
            
        });
        */
    }
    
}
