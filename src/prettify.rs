#![allow(dead_code)]
// Text Styles
pub static RESET: &str = "\x1b[0m";
pub static BOLD: &str = "\x1b[1m";
pub static DIM: &str = "\x1b[2m";
pub static ITALIC: &str = "\x1b[3m";
pub static UNDERLINE: &str = "\x1b[4m";
pub static BLINK: &str = "\x1b[5m";
pub static REVERSE: &str = "\x1b[7m";
pub static HIDDEN: &str = "\x1b[8m";
pub static STRIKETHROUGH: &str = "\x1b[9m";

// Standard Text Colors
pub static BLACK: &str = "\x1b[30m";
pub static RED: &str = "\x1b[31m";
pub static GREEN: &str = "\x1b[32m";
pub static YELLOW: &str = "\x1b[33m";
pub static BLUE: &str = "\x1b[34m";
pub static MAGENTA: &str = "\x1b[35m";
pub static CYAN: &str = "\x1b[36m";
pub static WHITE: &str = "\x1b[37m";

// Bright Text Colors
pub static BRIGHT_BLACK: &str = "\x1b[90m";
pub static BRIGHT_RED: &str = "\x1b[91m";
pub static BRIGHT_GREEN: &str = "\x1b[92m";
pub static BRIGHT_YELLOW: &str = "\x1b[93m";
pub static BRIGHT_BLUE: &str = "\x1b[94m";
pub static BRIGHT_MAGENTA: &str = "\x1b[95m";
pub static BRIGHT_CYAN: &str = "\x1b[96m";
pub static BRIGHT_WHITE: &str = "\x1b[97m";

// Standard Background Colors
pub static BG_BLACK: &str = "\x1b[40m";
pub static BG_RED: &str = "\x1b[41m";
pub static BG_GREEN: &str = "\x1b[42m";
pub static BG_YELLOW: &str = "\x1b[43m";
pub static BG_BLUE: &str = "\x1b[44m";
pub static BG_MAGENTA: &str = "\x1b[45m";
pub static BG_CYAN: &str = "\x1b[46m";
pub static BG_WHITE: &str = "\x1b[47m";

// Bright Background Colors
pub static BG_BRIGHT_BLACK: &str = "\x1b[100m";
pub static BG_BRIGHT_RED: &str = "\x1b[101m";
pub static BG_BRIGHT_GREEN: &str = "\x1b[102m";
pub static BG_BRIGHT_YELLOW: &str = "\x1b[103m";
pub static BG_BRIGHT_BLUE: &str = "\x1b[104m";
pub static BG_BRIGHT_MAGENTA: &str = "\x1b[105m";
pub static BG_BRIGHT_CYAN: &str = "\x1b[106m";
pub static BG_BRIGHT_WHITE: &str = "\x1b[107m";