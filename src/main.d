
/************************************************************************
Project: nn
Author: Marus Muja (2007)

*************************************************************************/
module main;

import std.stdio;

import util.random;
import util.logger;
import commands.all;


/** 
	Program entry point 
*/
void main(char[][] args)
{
	Logger.enableLevel(Logger.ERROR);
	// don't use garbage collector... manage memory manually
	std.gc.disable();
	
	if (args.length==1) {
		execute_command("help",args[0..1]);
		return;
	}
	
	int index = 1;
	if ( index<args.length || !is_command(args[index]) ) {
		if (args[index]=="help" && index+1==args.length) {
			execute_command("help",args[0..1]);
		}
		else {
			execute_command(args[index],args[index+1..$]);
		}
	} 
	else {
		execute_command("help",args[0..1]);
	}
	
	return 0;	
}



