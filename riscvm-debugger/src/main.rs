use std::io;

use ratatui::{
    crossterm::event::{self, KeyCode, KeyEventKind, KeyModifiers},
    prelude::*,
    style::Stylize,
    widgets::{Block, Borders, Cell, Padding, Paragraph, Row, Table, TableState, Wrap},
    DefaultTerminal,
};
use riscvm_debugger::debugger::{load_debugger_from_path, CommandOutput, Debugger};
use tui_popup::Popup;
use tui_prompts::{Prompt, State, TextPrompt, TextState};

#[derive(PartialEq, Eq)]
enum InputMode {
    Insert,
    Normal,
}

struct App<'a> {
    input: TextState<'a>,
    input_mode: InputMode,
    show_popup: bool,
    popup: Popup<'a, Text<'a>>,
    entries: Vec<String>,
    entry_idx: usize,
    should_quit: bool,
}

impl<'a> App<'a> {
    fn new() -> App<'a> {
        let mut input = TextState::new();
        input.focus();
        App {
            input,
            input_mode: InputMode::Normal,
            show_popup: false,
            popup: Popup::new(Text::from("")),
            entries: Vec::new(),
            entry_idx: 0,
            should_quit: false,
        }
    }

    fn show_message(&mut self, title: &'static str, message: impl Into<String>, color: Color) {
        self.popup = Popup::new(Text::from(message.into()))
            .title(title)
            .style(Style::new().fg(color).bg(Color::from_u32(0x202436)));
        self.show_popup = true;
    }
}

fn main() -> io::Result<()> {
    let cli = match parse_cli(std::env::args().skip(1)) {
        Ok(ParseResult::Run(cli)) => cli,
        Ok(ParseResult::Help) => {
            println!("{}", usage());
            return Ok(());
        }
        Err(error) => {
            eprintln!("{error}");
            eprintln!();
            eprintln!("{}", usage());
            std::process::exit(2);
        }
    };
    let debugger = load_debugger_from_path(&cli.path, cli.guest_args)?;

    if cli.batch {
        return run_batch(debugger, cli.commands);
    }

    let mut term = ratatui::init();
    term.clear()?;
    let app_result = run(term, debugger, App::new());
    ratatui::restore();
    app_result
}

#[derive(Debug, PartialEq, Eq)]
struct Cli {
    path: String,
    guest_args: Vec<String>,
    batch: bool,
    commands: Vec<String>,
}

#[derive(Debug)]
enum ParseResult {
    Run(Cli),
    Help,
}

fn parse_cli(args: impl IntoIterator<Item = String>) -> Result<ParseResult, String> {
    let mut batch = false;
    let mut commands = Vec::new();
    let mut path = None;
    let mut guest_args = Vec::new();
    let mut args = args.into_iter();

    while let Some(arg) = args.next() {
        if path.is_some() {
            guest_args.push(arg);
            guest_args.extend(args);
            break;
        }

        match arg.as_str() {
            "-h" | "--help" => return Ok(ParseResult::Help),
            "--batch" => batch = true,
            "-ex" | "--execute" | "--command" => {
                let Some(command) = args.next() else {
                    return Err(format!("{arg} requires a debugger command"));
                };
                batch = true;
                commands.push(command);
            }
            "--" => {
                let Some(binary) = args.next() else {
                    return Err("-- requires a binary path".to_string());
                };
                path = Some(binary);
                guest_args.extend(args);
                break;
            }
            _ if arg.starts_with('-') => return Err(format!("unknown option: {arg}")),
            _ => path = Some(arg),
        }
    }

    let Some(path) = path else {
        return Err("missing binary path".to_string());
    };

    Ok(ParseResult::Run(Cli {
        path,
        guest_args,
        batch,
        commands,
    }))
}

fn run_batch(mut debugger: Debugger, commands: Vec<String>) -> io::Result<()> {
    let commands = if commands.is_empty() {
        vec!["status".to_string()]
    } else {
        commands
    };

    for command in commands {
        println!("riscvm-debugger> {command}");
        let output = debugger.execute_command(&command);
        println!("{}", output.message);
        if output.should_quit {
            break;
        }
    }
    Ok(())
}

fn usage() -> &'static str {
    "Usage:
  riscvm-debugger <binary> [guest-args...]
  riscvm-debugger --batch -ex <command> [-ex <command>...] <binary> [guest-args...]

Options:
  --batch                 run commands without starting the TUI
  -ex, --execute <cmd>    execute one debugger command; implies --batch
  -h, --help              show this help"
}

fn run(mut term: DefaultTerminal, mut debugger: Debugger, mut app: App) -> io::Result<()> {
    let mut table_state = TableState::default();
    table_state.select_first();

    loop {
        if app.should_quit {
            return Ok(());
        }

        let instructions = debugger
            .disassemble_from(debugger.pc(), 50)
            .unwrap_or_else(|_| Vec::new());
        let rows = instructions
            .iter()
            .map(|instruction| {
                let marker = if debugger.breakpoints().contains(&instruction.address) {
                    "B"
                } else {
                    ""
                };
                Row::new(vec![
                    Cell::from(marker),
                    Cell::from(format!("0x{:016x}", instruction.address)),
                    Cell::from(format!("0x{:08x}", instruction.opcode)),
                    Cell::from(instruction.text.clone()),
                ])
            })
            .collect::<Vec<Row>>();

        term.draw(|frame| {
            draw(frame, &debugger, &mut app, &mut table_state, rows.clone());
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
                            KeyCode::Char('q') => return Ok(()),
                            KeyCode::Char('n') | KeyCode::Char('s') => {
                                let stop = debugger.step(1);
                                app.show_message("STEP", stop.message(), Color::Cyan);
                                table_state.select_first();
                            }
                            KeyCode::Char('c') => {
                                let stop = debugger.continue_execution();
                                app.show_message("CONTINUE", stop.message(), Color::Yellow);
                                table_state.select_first();
                            }
                            KeyCode::Char('i') | KeyCode::Char(':') => {
                                app.input_mode = InputMode::Insert;
                            }
                            KeyCode::Down => table_state.select_next(),
                            KeyCode::Up => table_state.select_previous(),
                            KeyCode::Enter => {
                                if let Some(selected) = table_state.selected() {
                                    if let Some(instruction) = instructions.get(selected) {
                                        let output = if debugger.pc() == instruction.address {
                                            CommandOutput {
                                                message: debugger.step(1).message(),
                                                should_quit: false,
                                            }
                                        } else {
                                            CommandOutput {
                                                message: debugger
                                                    .run_until_pc(instruction.address)
                                                    .message(),
                                                should_quit: false,
                                            }
                                        };
                                        apply_command_output(&mut app, output);
                                        table_state.select_first();
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                InputMode::Insert => {
                    if key.kind == KeyEventKind::Press {
                        match (key.code, key.modifiers) {
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) | (KeyCode::Esc, _) => {
                                app.input_mode = InputMode::Normal;
                            }
                            (KeyCode::Enter, _) => {
                                let cmd = app.input.value().to_string();
                                push_history(&mut app, &cmd);
                                let output = debugger.execute_command(&cmd);
                                app.input_mode = InputMode::Normal;
                                app.input.value_mut().clear();
                                app.input.move_start();
                                apply_command_output(&mut app, output);
                                table_state.select_first();
                            }
                            (KeyCode::Up, _) => history_prev(&mut app),
                            (KeyCode::Down, _) => history_next(&mut app),
                            _ => app.input.handle_key_event(key),
                        }
                    }
                }
            }
        }
    }
}

fn draw(
    frame: &mut Frame<'_>,
    debugger: &Debugger,
    app: &mut App<'_>,
    table_state: &mut TableState,
    rows: Vec<Row<'_>>,
) {
    let show_popup = app.show_popup;
    let style = if show_popup {
        Style::new().fg(Color::from_u32(0x555555))
    } else {
        Style::new()
    };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(1)])
        .split(frame.area());

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Length(43), Constraint::Min(60)])
        .split(root[0]);

    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Min(10),
        ])
        .split(body[0]);

    frame.render_widget(status_panel(debugger).style(style), left[0]);
    frame.render_widget(debugger_lists(debugger).style(style), left[1]);
    frame.render_widget(register_table(debugger).style(style), left[2]);
    frame.render_stateful_widget(instruction_table(rows).style(style), body[1], table_state);

    if app.input_mode == InputMode::Insert {
        TextPrompt::from(":").draw(frame, root[1], &mut app.input);
    } else {
        let hint =
            Paragraph::new("n/s step | c continue | enter run to selected | : command | q quit")
                .style(Style::new().fg(Color::DarkGray));
        frame.render_widget(hint, root[1]);
    }

    if show_popup {
        frame.render_widget(&app.popup, frame.area());
    }
}

fn status_panel(debugger: &Debugger) -> Paragraph<'static> {
    let stop = debugger
        .last_stop()
        .map_or_else(|| "ready".to_string(), |stop| stop.message());
    Paragraph::new(format!(
        "pc: 0x{:016x}\nbreakpoints: {}\nwatchpoints: {}\n{}",
        debugger.pc(),
        debugger.breakpoints().len(),
        debugger.watchpoints().len(),
        stop
    ))
    .block(
        Block::new()
            .title_top(Line::from("STATUS").centered())
            .borders(Borders::ALL)
            .padding(Padding::horizontal(1)),
    )
    .wrap(Wrap { trim: true })
}

fn debugger_lists(debugger: &Debugger) -> Paragraph<'static> {
    let breakpoints = if debugger.breakpoints().is_empty() {
        "breakpoints: none".to_string()
    } else {
        format!(
            "breakpoints: {}",
            debugger
                .breakpoints()
                .iter()
                .map(|address| format!("0x{address:016x}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let watchpoints = if debugger.watchpoints().is_empty() {
        "watchpoints: none".to_string()
    } else {
        format!(
            "watchpoints: {}",
            debugger
                .watchpoints()
                .iter()
                .map(|watch| format!("#{}", watch.id))
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    Paragraph::new(format!("{breakpoints}\n{watchpoints}"))
        .block(
            Block::new()
                .title_top(Line::from("STOPS").centered())
                .borders(Borders::ALL)
                .padding(Padding::horizontal(1)),
        )
        .wrap(Wrap { trim: true })
}

fn register_table(debugger: &Debugger) -> Table<'static> {
    let reg_widths = [Constraint::Length(7), Constraint::Length(20)];
    let mut rows = (0..32)
        .map(|index| {
            Row::new(vec![
                Cell::from(format!("x{index}:")),
                Cell::from(format!("0x{:016x}", debugger.cpu().registers[index])),
            ])
        })
        .collect::<Vec<Row>>();

    rows.push(Row::new(vec![
        Cell::from("pc:").fg(Color::from_u32(0x999cf0)),
        Cell::from(format!("0x{:016x}", debugger.pc())).fg(Color::from_u32(0x999cf0)),
    ]));

    Table::new(rows, reg_widths).block(
        Block::new()
            .title_top(Line::from("REGISTERS").centered())
            .borders(Borders::ALL)
            .padding(Padding::new(2, 1, 1, 1)),
    )
}

fn instruction_table(rows: Vec<Row<'_>>) -> Table<'_> {
    let widths = [
        Constraint::Length(2),
        Constraint::Length(20),
        Constraint::Length(12),
        Constraint::Min(24),
    ];

    Table::new(rows, widths)
        .header(
            Row::new(vec![
                Cell::from(""),
                Cell::from("Location").style(Style::new().fg(Color::Red)),
                Cell::from("Opcode").style(Style::new().fg(Color::Red)),
                Cell::from("Instruction").style(Style::new().fg(Color::Red)),
            ])
            .on_dark_gray(),
        )
        .block(
            Block::new()
                .title_top(Line::from("INSTRUCTIONS").centered())
                .borders(Borders::ALL),
        )
        .row_highlight_style(Style::new().bg(Color::from_u32(0x32502c)))
        .highlight_symbol(">>")
}

fn apply_command_output(app: &mut App<'_>, output: CommandOutput) {
    app.should_quit = output.should_quit;
    let color = if output.should_quit {
        Color::Yellow
    } else if output.message.contains("unknown")
        || output.message.contains("invalid")
        || output.message.contains("requires")
        || output.message.contains("failed")
    {
        Color::Red
    } else {
        Color::White
    };
    app.show_message("COMMAND", output.message, color);
}

fn push_history(app: &mut App<'_>, cmd: &str) {
    if cmd.trim().is_empty() {
        return;
    }
    if app.entries.last().is_none_or(|entry| entry != cmd) {
        app.entries.push(cmd.to_string());
    }
    app.entry_idx = app.entries.len();
}

fn history_prev(app: &mut App<'_>) {
    if app.entries.is_empty() {
        return;
    }
    app.entry_idx = app.entry_idx.saturating_sub(1);
    if let Some(entry) = app.entries.get(app.entry_idx) {
        *app.input.value_mut() = entry.clone();
    }
}

fn history_next(app: &mut App<'_>) {
    if app.entries.is_empty() {
        return;
    }
    if app.entry_idx + 1 >= app.entries.len() {
        app.entry_idx = app.entries.len();
        app.input.value_mut().clear();
        return;
    }
    app.entry_idx += 1;
    if let Some(entry) = app.entries.get(app.entry_idx) {
        *app.input.value_mut() = entry.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<ParseResult, String> {
        parse_cli(args.iter().map(|arg| arg.to_string()))
    }

    #[test]
    fn parses_interactive_binary_and_guest_args() {
        let ParseResult::Run(cli) = parse(&["program", "one", "two"]).unwrap() else {
            panic!("expected runnable cli");
        };

        assert_eq!(
            cli,
            Cli {
                path: "program".to_string(),
                guest_args: vec!["one".to_string(), "two".to_string()],
                batch: false,
                commands: Vec::new(),
            }
        );
    }

    #[test]
    fn parses_batch_commands_before_binary_path() {
        let ParseResult::Run(cli) = parse(&[
            "--batch",
            "-ex",
            "break pc",
            "--execute",
            "continue limit 5",
            "program",
            "guest",
        ])
        .unwrap() else {
            panic!("expected runnable cli");
        };

        assert_eq!(cli.path, "program");
        assert_eq!(cli.guest_args, vec!["guest"]);
        assert!(cli.batch);
        assert_eq!(cli.commands, vec!["break pc", "continue limit 5"]);
    }

    #[test]
    fn parses_double_dash_before_binary_path() {
        let ParseResult::Run(cli) = parse(&["--batch", "-ex", "status", "--", "-program"]).unwrap()
        else {
            panic!("expected runnable cli");
        };

        assert_eq!(cli.path, "-program");
        assert!(cli.guest_args.is_empty());
        assert!(cli.batch);
    }

    #[test]
    fn reports_missing_execute_argument() {
        assert_eq!(
            parse(&["-ex"]).unwrap_err(),
            "-ex requires a debugger command"
        );
    }
}
