use crate::cli::{RvnCli, Command };
//use crate::commands::info;
//use anyhow::Result;
use crate::commands::list::run_list;

pub fn dispatch(cli: RvnCli) -> anyhow::Result<()> 
{
    println!("dispatch");
    match cli.command {
        Command::List { ref file, .. } => { 
            run_list(file)?; 
        }
        Command::ForwardSimple { file, q, k, v } => { 
            println!("run forward simple");
        }
        Command::Info(cmd) => {
        }
        Command::Dump { file, name, format, output } => { 
            println!("run dump");
        }
        Command::Analyze { file, output } => { 
            println!("run analyze");
        }
        Command::Validate { file, profile, output } => { 
            println!("run validate");
        }
        Command::Forward(_) => { 
            println!("[TODO] run forward not implemented yet");
        }
        Command::Diff { file_a, file_b, .. } => { 
            println!("[TODO] Diff '{}' vs '{}' not implemented yet", file_a, file_b);
        } 
        Command::Profile { file, .. } => {
            println!("[TODO] Profile not implemented yet for file: {}", file);
        }
        Command::Watch { file, .. } => {
            println!("[TODO] Watch not yet implemented")
        }
        Command::WatchPerf { file, .. } => {
            println!("[TODO] WatchPerf not implemented yet for file: {}", file);
        }
        Command::Debug { file, threads, output, compat } => {
               
        }
        Command::DecodeTest { file, name, verbose, json, fail_on_anomaly } => {
        
        }
        // handle all other commands
    }

    Ok(())
}


