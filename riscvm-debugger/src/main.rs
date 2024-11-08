use std::{
    collections::HashSet,
    fs::File,
    io::{self, Read},
};

use layout::Flex;
use ratatui::{
    crossterm::{
        self,
        event::{self, KeyCode, KeyEventKind, KeyModifiers},
    },
    prelude::*,
    style::Stylize,
    widgets::{Block, Borders, Cell, List, Padding, Paragraph, Row, Table, TableState},
    DefaultTerminal,
};
use riscvm_core::cpu::{RV64GCInstruction, RV64GC};
use tui_popup::Popup;
use tui_prompts::{Prompt, State, TextPrompt, TextState};

#[derive(PartialEq, Eq)]
enum InputMode {
    Insert,
    Normal,
}

struct App<'a> {
    pub input: TextState<'a>,
    pub input_mode: InputMode,
    pub show_popup: bool,
    pub popup: Popup<'a, Text<'a>>,
    pub entries: Vec<String>,
    pub entry_idx: usize,

    pub breakpoints: Vec<u64>,
}

impl<'a> App<'a> {
    pub fn new() -> App<'a> {
        let mut input = TextState::new();
        input.focus();
        App {
            input,
            input_mode: InputMode::Normal,
            show_popup: false,
            popup: Popup::new(Text::from("")),
            entries: Vec::new(),
            entry_idx: 0,
            breakpoints: vec![],
        }
    }
}

fn main() -> io::Result<()> {
    let mut cpu = RV64GC::new();
    let Some(path) = std::env::args().nth(1) else {
        panic!("No elf file passed!");
    };

    let mut buf = Vec::new();
    File::open(path).unwrap().read_to_end(&mut buf).unwrap();

    cpu.load_elf(buf).unwrap();

    let app = App::new();

    let mut term = ratatui::init();
    term.clear()?;
    let app_result = run(term, cpu, app);
    ratatui::restore();
    app_result
}

fn update_ins_table(cpu: &RV64GC, ins_count: usize) -> Vec<(u64, RV64GCInstruction)> {
    let mut ins_vec = Vec::new();

    let mut pc = cpu.registers[32];
    for _ in 0..ins_count {
        let ins = cpu.ram.read_word(pc).unwrap_or(0x00000013);
        let dec = cpu.find_instruction(ins);
        ins_vec.push((pc, dec));

        if ins & 3 != 3 {
            pc += 2;
        } else {
            pc += 4;
        }
    }

    ins_vec
}

fn jump_to_point(
    cpu: &mut RV64GC,
    app: &mut App,
    table_state: &TableState,
    ins_vec: &[(u64, RV64GCInstruction)],
) {
    let Some(selected_ins) = table_state.selected() else {
        return;
    };

    let selection = ins_vec[selected_ins];

    if cpu.registers[32] == selection.0 {
        cpu.step();
        return;
    }

    while cpu.registers[32] != selection.0 {
        if app.breakpoints.contains(&cpu.registers[32]) {
            app.popup = Popup::new(Text::from(format!(
                "Breakpoint at {:x} hit!",
                cpu.registers[32]
            )))
            .title("BREAKPOINT")
            .style(Style::new().fg(Color::Yellow).bg(Color::from_u32(0x3b3f63)));
            app.show_popup = true;
            return;
        }

        cpu.step();
    }
}

fn handle_cmd<'a>(app: &mut App, cpu: &mut RV64GC, cmd: String) -> Popup<'a, Text<'a>> {
    let split_cmds = cmd.split_whitespace().collect::<Vec<_>>();

    let bg_color = Color::from_u32(0x3b3f63);
    let style = Style::new().bg(bg_color);

    let err_popup = |text| {
        Popup::new(Text::from(text).centered())
            .style(Style::new().fg(Color::Red).bg(bg_color))
            .title("Command")
    };

    if split_cmds.is_empty() {
        return err_popup("Invalid Command!");
    }

    match split_cmds[0] {
        "mem" => {
            let (Some(oper), Some(str_addr)) = (split_cmds.get(1), split_cmds.get(2)) else {
                return err_popup("Missing operation and/or address!");
            };
            match *oper {
                "read" => {
                    let offset = split_cmds
                        .get(3)
                        .and_then(|s| s.parse::<i64>().ok())
                        .unwrap_or(0);

                    let addr = if str_addr.starts_with("x") {
                        str_addr
                            .trim_start_matches("x")
                            .parse::<u8>()
                            .map(|i| cpu.registers[&i])
                    } else {
                        u64::from_str_radix(str_addr.trim_start_matches("0x"), 16)
                    };

                    let Ok(addr) = addr else {
                        return err_popup("Invalid address!\nPlease use a hex address");
                    };

                    let value = cpu.ram.read_doubleword(addr.wrapping_add_signed(offset));

                    let Ok(value) = value else {
                        return err_popup("This addresss has not been mapped!");
                    };

                    Popup::new(
                        Text::from(format!("{str_addr} + {offset} -> 0x{value:016x}")).centered(),
                    )
                    .title("MEMORY")
                    .style(style)
                }
                _ => err_popup("Invalid Command!"),
            }
        }

        "reset" => {
            cpu.reset();
            Popup::new(Text::from("cpu has been reset!").centered())
                .title("CPU")
                .style(style)
        }

        "breakpoint" | "b" => {
            let (Some(oper), Some(addr)) = (
                split_cmds.get(1),
                split_cmds
                    .get(2)
                    .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
            ) else {
                return err_popup("Missing operation and/or address!");
            };

            match *oper {
                "set" | "s" => {
                    app.breakpoints.push(addr);

                    Popup::new(
                        Text::from(format!("Breakpoint at {:08x} is now set!", addr)).centered(),
                    )
                    .title("BREAKPOINT")
                    .style(style)
                }

                "clear" | "c" => {
                    app.breakpoints.clear();

                    Popup::new(Text::from("All breakpoints cleared!").centered())
                        .title("BREAKPOINT")
                        .style(style)
                }

                "delete" | "d" => {
                    let result = app.breakpoints.iter().position(|i| *i == addr);

                    if let Some(pos) = result {
                        app.breakpoints.remove(pos);

                        Popup::new(
                            Text::from(format!("Breakpoint at {addr:08x} removed!")).centered(),
                        )
                        .title("BREAKPOINT")
                        .style(style)
                    } else {
                        err_popup("No breakpoint found!")
                    }
                }

                _ => err_popup("Invalid operation!"),
            }
        }

        "cont" | "c" => {
            if split_cmds.len() == 1 {
                while !app.breakpoints.contains(&cpu.registers[32]) {
                    cpu.step();
                }

                return Popup::new(Text::from(format!(
                    "Breakpoint at {:x} hit!",
                    cpu.registers[32]
                )))
                .title("BREAKPOINT")
                .style(Style::new().fg(Color::Yellow).bg(Color::from_u32(0x3b3f63)));
            }

            let (Some(oper), Some(str_addr)) = (split_cmds.get(1), split_cmds.get(2)) else {
                return err_popup("Missing operation and/or address!");
            };

            match *oper {
                "unchanged" => {
                    if str_addr.starts_with("x") {
                        let Ok(reg_num) = str_addr.trim_start_matches("x").parse::<u8>() else {
                            return err_popup("Invalid Register!");
                        };

                        let og_copy = cpu.registers[&reg_num];
                        while cpu.registers[&reg_num] == og_copy {
                            cpu.step();
                        }
                    } else {
                        let Ok(addr) = u64::from_str_radix(str_addr.trim_start_matches("0x"), 16)
                        else {
                            return err_popup("Invalid Address!");
                        };

                        let Ok(og_copy) = cpu.ram.read_doubleword(addr) else {
                            return err_popup("Address has not been mapped!");
                        };

                        while og_copy == cpu.ram.read_doubleword(addr).unwrap() {
                            cpu.step();
                        }
                    };
                    Popup::new(Text::from(format!("at {str_addr} now!")).centered()).style(style)
                }
                "to" => {
                    let addr = u64::from_str_radix(str_addr.trim_start_matches("0x"), 16);

                    let Ok(addr) = addr else {
                        return err_popup("Invalid address!\nPlease use a hex address");
                    };

                    while cpu.registers[32] != addr {
                        cpu.step();
                    }

                    Popup::new(Text::from(format!("at {str_addr} now!")).centered()).style(style)
                }
                _ => err_popup("Invalid Operation!"),
            }
        }
        _ => err_popup("Invalid Command!"),
    }
}

fn run(mut term: DefaultTerminal, mut cpu: RV64GC, mut app: App) -> io::Result<()> {
    let mut table_state = TableState::default();
    table_state.select_first();

    loop {
        let ins_rows = update_ins_table(&cpu, 50);
        let rows = ins_rows
            .clone()
            .into_iter()
            .map(|i| Row::new(vec![format!("0x{:08x}", i.0), format!("{}", i.1)]))
            .collect::<Vec<Row>>();

        let row_widths = [Constraint::Length(20), Constraint::Length(20)];
        let reg_widths = [Constraint::Length(5), Constraint::Percentage(100)];

        term.draw(|frame| {
            let show_popup = app.show_popup;
            let style = if show_popup {
                Style::new().fg(Color::from_u32(0x555555))
            } else {
                Style::new()
            };

            let vert_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vec![Constraint::Percentage(100), Constraint::Min(1)])
                .split(frame.area());

            let layout = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(vec![Constraint::Percentage(35), Constraint::Percentage(65)])
                .split(vert_layout[0]);

            let mut reg_items = (0..32)
                .map(|i| {
                    Row::new(vec![
                        Text::from(format!("x{i}:")),
                        Text::from(format!("0x{:08x}", cpu.registers[i])),
                    ])
                })
                .collect::<Vec<Row>>();

            reg_items.push(Row::new(vec![
                Text::from("pc:").fg(Color::from_u32(0x999cf0)),
                Text::from(format!("0x{:08x}", cpu.registers[32])).fg(Color::from_u32(0x999cf0)),
            ]));

            let reg_list = Table::new(reg_items, reg_widths).block(
                Block::new()
                    .title_top(Line::from("REGISTERS").centered())
                    .borders(Borders::ALL)
                    .padding(Padding::new(4, 4, 1, 1))
                    .style(style),
            );

            let table = Table::new(rows, row_widths)
                .header(
                    Row::new(vec![
                        Cell::from("Location").style(Style::new().fg(Color::Red)),
                        Cell::from("Instruction").style(Style::new().fg(Color::Red)),
                    ])
                    .on_dark_gray(),
                )
                .block(
                    Block::new()
                        .title_top(Line::from("INSTRUCTIONS").centered())
                        .borders(Borders::ALL)
                        .style(style),
                )
                .flex(Flex::SpaceAround)
                .row_highlight_style(Style::new().bg(Color::from_u32(0x32502c)))
                .highlight_symbol(">>");

            frame.render_widget(reg_list, layout[0]);
            frame.render_stateful_widget(table, layout[1], &mut table_state);

            if app.input_mode == InputMode::Insert {
                TextPrompt::from(":").draw(frame, vert_layout[1], &mut app.input);
            }

            if show_popup {
                frame.render_widget(&app.popup, frame.area());
            }
        })?;

        if let event::Event::Key(key) = event::read()? {
            match app.input_mode {
                InputMode::Normal => {
                    if app.show_popup {
                        app.show_popup = false;
                        continue;
                    }
                    if key.kind == KeyEventKind::Press {
                        match key.code {
                            KeyCode::Char('q') => {
                                return Ok(());
                            }
                            KeyCode::Char('n') => {
                                cpu.step();

                                table_state.select_first();
                            }
                            KeyCode::Char('i') => app.input_mode = InputMode::Insert,
                            KeyCode::Char(':') => app.input_mode = InputMode::Insert,
                            KeyCode::Down => table_state.select_next(),
                            KeyCode::Up => table_state.select_previous(),
                            KeyCode::Enter => {
                                jump_to_point(&mut cpu, &mut app, &table_state, &ins_rows)
                            }
                            _ => {}
                        }
                    }
                }
                InputMode::Insert => {
                    if key.kind == KeyEventKind::Press {
                        match (key.code, key.modifiers) {
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                app.input_mode = InputMode::Normal
                            }

                            (KeyCode::Enter, _) => {
                                let cmd = app.input.value().to_string();
                                let popup = handle_cmd(&mut app, &mut cpu, cmd);
                                app.input_mode = InputMode::Normal;
                                app.input.value_mut().clear();
                                app.input.move_start();

                                if let Some(entry) = app.entries.last() {
                                    if app.input.value() != entry {
                                        app.entries.push(app.input.value().to_string());
                                        app.entry_idx += 1;
                                    }
                                } else {
                                    app.entries.push(app.input.value().to_string());
                                    app.entry_idx += 1;
                                }

                                app.popup = popup;
                                app.show_popup = true;
                            }

                            (KeyCode::Up, _) => {
                                if let Some(entry) = app.entries.get(app.entry_idx) {
                                    *app.input.value_mut() = entry.clone();
                                }

                                if app.entry_idx != 0 {
                                    app.entry_idx -= 1;
                                }
                            }

                            (KeyCode::Down, _) => {
                                if let Some(entry) = app.entries.get(app.entry_idx) {
                                    *app.input.value_mut() = entry.clone();
                                } else {
                                    app.input.value_mut().clear();
                                }

                                if app.entry_idx < app.entries.len() {
                                    app.entry_idx += 1;
                                }
                            }

                            (KeyCode::Esc, _) => app.input_mode = InputMode::Normal,
                            _ => app.input.handle_key_event(key),
                        }
                    }
                }
            }
        }
    }
}
