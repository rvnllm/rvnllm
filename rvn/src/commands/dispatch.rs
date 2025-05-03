use crate::cli::{RvnCli, Command };
//use crate::commands::info;
//use anyhow::Result;
use crate::commands::list::run_list;

pub fn dispatch(cli: RvnCli) -> anyhow::Result<()> 
{
    println!("dispatch");
    match cli.command {
        Command::List { file, output: _ } => {
            run_list(file)?; 
        }
        Command::ForwardSimple { 
            file: _, 
            q: _, 
            k: _, 
            v: _ } => { 
            println!("run forward simple");
        }
        Command::Info(_) => {
            println!("[TODO] Info not implemented yet");
        }
        Command::Dump { 
            file: _, 
            name: _, 
            format: _,
            output: _ } => { 
            println!("run dump");
        }
        Command::Analyze { file: _, output: _ } => { 
            println!("run analyze");
        }
        Command::Validate { 
            file: _, 
            profile: _, 
            output: _ } => {
                println!("run validate");
        }
        Command::Forward(_) => {
            println!("[TODO] run forward not implemented yet");
        }
        Command::Diff { 
            file_a, 
            file_b, 
            output: _ } => {
            println!("[TODO] Diff '{}' vs '{}' not implemented yet", file_a, file_b);
        } 
        Command::Profile { 
            file, 
            device: _, 
            tokens: _, 
            cache_mode: _, 
            output: _ } => {
            println!("[TODO] Profile not implemented yet for file: {}", file);
        }
        Command::Watch { 
            file: _, 
            profile: _, 
            dummy_forward: _, 
            check_tokenizer: _, 
            check_entropy: _, 
            scan_triggers: _, 
            verbose: _, 
            output: _ } => {

            println!("[TODO] Watch not yet implemented");
        }
        Command::WatchPerf { 
            file, 
            input: _, 
            metrics: _, 
            preset: _, 
            device: _, 
            output: _ } => {

            println!("[TODO] WatchPerf not implemented yet for file: {}", file);
        }
        Command::Debug { 
            file, 
            threads: _, 
            output: _, 
            compat: _ } => {   
            println!("[TODO] Debug not implemented yet for file: {}", file);
        }
        Command::DecodeTest { 
            file, name: _, verbose: _, json: _, fail_on_anomaly: _ } => {    
            println!("[TODO] DecodeTest not implemented yet for file: {}", file);
        }
        // handle all other commands
    }

    Ok(())
}


