#![feature(scoped_threads)]

mod cpu6502;
mod sim;
mod types;

use cpu6502::CPU6502;
use std::fs::File;
use std::io::{ Read, Seek, SeekFrom };
use std::time::*;

pub fn print_status(cpu: &CPU6502, memory: &[u8])
{

        println!("{}\t\tNV-BDIZC\tPC: 0x{:04x}\n{:02x} {:02x} {:02x}\t{:08b}\n{} 0x{:x}: {:x}\n", 
            if cpu.read_clk() == 0 { "LO" } else { "HI" },
            cpu.read_pc(),
            cpu.read_A(), cpu.read_X(), cpu.read_Y(),
            cpu.read_P(),
            if cpu.read_rw() { 'R' } else { 'W' },
            cpu.read_addr(),
            if cpu.read_rw() { memory[cpu.read_addr() as usize] } else { cpu.read_data()});
}

fn main() {

    let n = Instant::now();
    
    let mut memory = [0; 0x10000];

    let mut rom = File::open("6502_functional_test.bin")
        .expect("can't read rom");
    rom.seek(SeekFrom::Start(0));
    rom.read_exact(&mut memory[0x0..]);

    memory[0x2002] = 0xFF;

    println!("Rom loaded. Took {}", n.elapsed().as_micros());
    let n = Instant::now();

    let mut cpu = CPU6502::new(); 
    
    println!("CPU loaded and set up. Took {}", n.elapsed().as_micros());
    let n = Instant::now();

    cpu.initialize();


    // TODO: rename this to step
    cpu.cpu_step();
    
    let n = Instant::now();

    const CYCLES: u32 = 100000;
    const THRESHOLD: u32 = 1000;

    let mut cycle = 0;

    let mut total_time = 0;

    memory[0xfffc] = 0x00;
    memory[0xfffd] = 0x04;

    for i in 0..CYCLES
    {
        cpu.cpu_step();
        //print_status(&cpu, &memory);

        let addr = cpu.read_addr() as usize;

        if cpu.read_clk() == 1
        {
            if cpu.read_rw()
            {    
                if addr == 0x3469
                {
                    println!("SUCCESS!");
                    break;
                }
                cpu.write_data(memory[addr]);
            }
            else
            {
                let data = cpu.read_data();
                memory[addr] = data;
                if addr == 0x200
                {
                    println!("=== Wrote {:X} to 0x{:X}", data, addr);
                }
            }
        }


        if cycle % 10000 == 0
        {
            println!("Current PC: 0x{:X} - Cycle {}", cpu.read_pc(), cycle/2);
        }

        cycle += 1;


    }
    println!();
    //println!("Average Cycle Time: {} us", total_time/CYCLES);

    let time = n.elapsed().as_micros();

    //print stack
    println!("{:x}", cpu.read_S());
    for i in 10..0x10
    {
        println!("{:02x}: {:x?}", i*0x10, &memory[0x100+i*0x10..0x100+(i+1)*0x10]);
    }

    println!("Ran {} cycles in {}: {} Hz", 8+CYCLES, time, ((8+CYCLES) as f32)/(time as f32) * 1000000f32 / 2.0);

    print_status(&cpu, &memory);
    // TODO: figure out how much slower memset over a bitmap is than
    //       removing and remaking the big ass vec
}
