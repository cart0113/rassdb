"""UI components for RASSDB using shellack."""

import signal
import sys
from pathlib import Path
from typing import Optional, Callable

from shellack import colors, gradients
from shellack.interface import Interface
from shellack.progress_bars import ProgressBar
from shellack.rowcomponents import Row
from shellack.terminal import Key, Modifiers
from shellack.textbox import BareTextBox
from shellack.textcomponents import Text


class IndexingProgress:
    """Progress display for file indexing using shellack."""

    def __init__(self, total_files: int, quiet: bool = False):
        """Initialize the indexing progress display.

        Args:
            total_files: Total number of files to index
            quiet: Whether to run in quiet mode (no UI)
        """
        self.total_files = total_files
        self.quiet = quiet
        self.current_file_idx = 0
        self.current_file_path = ""
        self.files_processed = 0
        self.chunks_processed = 0
        self._cancelled = False

        if not quiet:
            # Create interface
            self.tui = Interface(
                on_key_press_callback=self.handle_key_press,
                non_blocking=True,  # Non-blocking so indexing can continue
            )

            # Create progress gradient
            progress_gradient = gradients.LinearGradient(
                start_color=colors.BLUE,
                stop_color=colors.GREEN,
                steps=50,
                end_style="blunt",
            )

            # Create progress bar
            self.progress_bar = ProgressBar(
                left_bookend_text="[",
                right_bookend_text="]",
                fill_chars=["█"],
                tip_text="▶",
                empty_char="░",
                progress_gradient=progress_gradient,
                right_text="percent",
                right_text_color=colors.GRAY_BLUE,
            )

            # Status text components
            self.status_text = Text("Initializing...", color=colors.GRAY_BLUE)
            self.file_text = Text("", color=colors.WHITE)
            self.stats_text = Text("", color=colors.GRAY)

            # Build UI layout
            self.status_row = Row([self.status_text])
            self.file_row = Row([Text("File: ", color=colors.GRAY), self.file_text])
            self.stats_row = Row([self.stats_text])
            self.progress_row = Row([self.progress_bar])

            # Help text
            self.help_text = Text(
                "Press Ctrl+C to cancel", color=colors.GRAY + colors.ITALIC
            )
            self.help_row = Row([self.help_text])

            # Add components to interface
            self.tui.component_map["status"] = self.status_row
            self.tui.component_map["file"] = self.file_row
            self.tui.component_map["progress"] = self.progress_row
            self.tui.component_map["stats"] = self.stats_row
            self.tui.component_map["help"] = self.help_row

            # Set up signal handler for Ctrl+C
            signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C signal."""
        self._cancelled = True
        if not self.quiet:
            self.status_text.text = "Cancelling..."
            self.status_text.color = colors.ORANGE
            self.tui.render()

    def handle_key_press(self, key: Key, modifiers: Modifiers) -> bool:
        """Handle keyboard input.

        Args:
            key: The key that was pressed
            modifiers: Active modifier keys

        Returns:
            True to continue running, False to exit
        """
        # Handle Ctrl+C
        if modifiers.ctrl and key.characters == "c":
            self._cancelled = True
            self.status_text.text = "Cancelling..."
            self.status_text.color = colors.ORANGE
            return True

        # Handle Ctrl+D
        if modifiers.ctrl and key.characters == "d":
            self._cancelled = True
            self.status_text.text = "Cancelling..."
            self.status_text.color = colors.ORANGE
            return True

        return True

    def start(self):
        """Start the progress display."""
        if not self.quiet:
            self.tui.start()
            self.status_text.text = "Indexing files..."
            self.status_text.color = colors.GREEN
            self.tui.render()

    def update_file(self, file_path: str, file_idx: int):
        """Update the current file being processed.

        Args:
            file_path: Path to the current file
            file_idx: Index of current file (0-based)
        """
        self.current_file_path = file_path
        self.current_file_idx = file_idx

        if not self.quiet:
            # Truncate long paths for display
            display_path = file_path
            if len(display_path) > 60:
                display_path = "..." + display_path[-57:]

            self.file_text.text = display_path

            # Update progress
            progress = (
                (file_idx / self.total_files) * 100 if self.total_files > 0 else 0
            )
            self.progress_bar.set_progress(progress)

            # Update stats
            self.stats_text.text = f"Files: {self.files_processed}/{self.total_files} | Chunks: {self.chunks_processed}"

            self.tui.render()

    def increment_chunks(self, num_chunks: int):
        """Increment the number of chunks processed.

        Args:
            num_chunks: Number of chunks to add
        """
        self.chunks_processed += num_chunks
        if not self.quiet:
            self.stats_text.text = f"Files: {self.files_processed}/{self.total_files} | Chunks: {self.chunks_processed}"
            self.tui.render()

    def complete_file(self):
        """Mark the current file as completed."""
        self.files_processed += 1
        if not self.quiet:
            progress = (
                (self.files_processed / self.total_files) * 100
                if self.total_files > 0
                else 0
            )
            self.progress_bar.set_progress(progress)
            self.stats_text.text = f"Files: {self.files_processed}/{self.total_files} | Chunks: {self.chunks_processed}"
            self.tui.render()

    def finish(self, success: bool = True):
        """Finish the progress display.

        Args:
            success: Whether indexing completed successfully
        """
        if not self.quiet:
            if success:
                self.status_text.text = "✓ Indexing complete!"
                self.status_text.color = colors.GREEN
                self.progress_bar.set_progress(100)
            else:
                self.status_text.text = "✗ Indexing cancelled"
                self.status_text.color = colors.RED

            self.file_text.text = ""
            self.help_text.text = "Press any key to exit"

            # Final render
            self.tui.render()

            # Wait for key press to exit
            self.tui.blocking = True
            self.tui.get_next_key()

            # Stop the interface
            self.tui.stop()

    def is_cancelled(self) -> bool:
        """Check if indexing was cancelled.

        Returns:
            True if cancelled
        """
        if not self.quiet:
            # Process any pending key events
            self.tui.process_events()
        return self._cancelled
