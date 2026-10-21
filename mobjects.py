import numpy as np
from manim import *

# cube in 3d space opengl interactive

class CubeFrame(VGroup):
    def __init__(self,
        side_length=2,
        thickness=0.08,
        color=WHITE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        h = side_length / 2
        corners = [-h, h]
        style = dict(fill_color=color, fill_opacity=1, stroke_width=0)

        def tube(dimensions, position):
            return Prism(dimensions=dimensions, **style).move_to(position)
        
        tube_side_length = side_length + thickness
        for y in corners:
            for z in corners:
                self.add(tube([tube_side_length, thickness, thickness], [0, y, z]))
        for x in corners:
            for z in corners:
                self.add(tube([thickness, tube_side_length, thickness], [x, 0, z]))
        for x in corners:
            for y in corners:
                self.add(tube([thickness, thickness, tube_side_length], [x, y, 0]))



class CubeLabel(VGroup):
    def __init__(self,
        value=1,
        height=0.5,
        color=WHITE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.value = float(value)
        self.label_height = height
        self.color = color
        rounded = round(self.value, 1)
        self.number = Text(f"{rounded:.1f}", color=self.color)
        self.number.scale_to_fit_height(self.label_height)
        self.add(self.number)
        self.face_camera()

    def face_camera(self):
        if config.renderer != RendererType.OPENGL:
            return self
        center = tuple(self.get_center())
        for mob in self.get_family():
            mob.fix_orientation()
            mob.fixed_orientation_center = center
        return self


class SetRowValues(Animation):
    def __init__(self, row, values, **kwargs):
        self.target_values = [float(value) for value in values]
        super().__init__(row, **kwargs)

    def begin(self):
        self.start_values = list(self.mobject.values)
        super().begin()

    def interpolate_mobject(self, alpha):
        alpha = self.rate_func(alpha)
        current = [
            start + (end - start) * alpha
            for start, end in zip(self.start_values, self.target_values)
        ]
        self.mobject.set_values(current)


class RowOfCubes(Group):
    def __init__(
        self,
        values,
        start=LEFT * 4,
        end=RIGHT * 4,
        spacing=0.3,
        show_labels=True,
        fill_color=BLUE,
        fill_opacity=0.9,
        opacity_by_value=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        values = [float(value) for value in values]
        peak = max(values) if values else 0
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        span = end - start
        length = float(np.linalg.norm(span))
        count = len(values)
        gaps = max(count - 1, 0)
        self.side_length = (length - gaps * spacing) / count if count else 0
        if count and self.side_length <= 0:
            raise ValueError("spacing is too large for the distance between start and end")

        self.values = values
        self.opacity_by_value = opacity_by_value
        self.fill_color = fill_color
        self.fill_opacity = fill_opacity
        self.show_labels = show_labels

        self.cubes = VGroup()
        for index, value in enumerate(values):
            if self.opacity_by_value:
                opacity = 0 if peak <= 0 else min(1, max(0, value / peak))
            else:
                opacity = self.fill_opacity
            cube = Cube(
                side_length=self.side_length,
                fill_color=self.fill_color,
                fill_opacity=opacity,
            )
            frame = CubeFrame(
                side_length=self.side_length,
                thickness=self.side_length * 0.04,
            )
            body = VGroup(cube, frame)
            body.move_to(RIGHT * (self.side_length / 2 + index * (self.side_length + spacing)))
            self.cubes.add(body)

        if length > 1e-8 and count:
            self._point_along(self.cubes, span / length)
            self.cubes.shift(start)

        self.add(self.cubes)
        self.labels = []
        if show_labels:
            for value, body in zip(values, self.cubes):
                self._add_label(value, body)

    def _add_label(self, value, body):
        cube = body[0]
        is_opaque_white = (
            np.allclose(color_to_rgb(cube.get_fill_color()), color_to_rgb(WHITE))
            and cube.get_fill_opacity() > 0.5
        )
        label = CubeLabel(
            value,
            height=self.side_length * 0.3,
            color=BLACK if is_opaque_white else WHITE,
        )
        label.tracked_body = body
        label.move_to(body.get_center()).face_camera()
        label.add_updater(lambda m: m.move_to(m.tracked_body.get_center()).face_camera())
        self.labels.append(label)
        self.add(label)
        return label

    def set_values(self, values, show_labels=None):
        if show_labels is not None:
            self.show_labels = show_labels

        values = [float(value) for value in values]
        self.values = values
        peak = max(values) if values else 0

        for value, body in zip(values, self.cubes):
            if self.opacity_by_value:
                opacity = 0 if peak <= 0 else min(1, max(0, value / peak))
            else:
                opacity = self.fill_opacity
            body[0].set_fill(color=self.fill_color, opacity=opacity)

            for label in self.labels:
                label.clear_updaters()
                self.remove(label)
            self.labels = []
        if self.show_labels:
            for value, body in zip(values, self.cubes):
                self._add_label(value, body)
        return self

    @override_animate(set_values)
    def _set_values_animation(self, values, anim_args=None):
        return SetRowValues(self, values, **(anim_args or {}))


    def _point_along(self, cubes, direction):
        axis = np.cross(RIGHT, direction)
        if np.linalg.norm(axis) < 1e-8:
            if direction[0] < 0:
                cubes.rotate(PI, axis=UP, about_point=ORIGIN)
            return
        angle = np.arccos(np.clip(float(np.dot(RIGHT, direction)), -1.0, 1.0))
        cubes.rotate(angle, axis=axis, about_point=ORIGIN)

    def set_fill_color(self, color):
        self.fill_color = color
        for body in self.cubes:
            body[0].set_fill(color=color)
        return self

    def set_cube_fill_opacity(self, index, opacity, frame_opacity=None, color=None):
        body = self.cubes[index]
        body[0].set_fill(
            color=self.fill_color if color is None else color,
            opacity=opacity,
        )
        if frame_opacity is not None:
            body[1].set_opacity(frame_opacity)
        return self

    def set_body_visible(self, index, visible, fill_opacity=0, color=None):
        if visible:
            self.set_cube_fill_opacity(
                index,
                fill_opacity,
                frame_opacity=1,
                color=color,
            )
        else:
            self.set_cube_fill_opacity(index, 0, frame_opacity=0)
            self._remove_label_for(self.cubes[index])
        return self

    def _remove_label_for(self, body):
        kept = []
        for label in self.labels:
            if label.tracked_body is body:
                label.clear_updaters()
                self.remove(label)
            else:
                kept.append(label)
        self.labels = kept
        return self

    def drop_labels(self):
        for label in list(self.labels):
            label.clear_updaters()
            self.remove(label)
        self.labels = []
        return self

    def clear_labels(self):
        self.drop_labels()
        self.show_labels = False
        return self

class SnapJump(Animation):
    def __init__(self, conv, step, **kwargs):
        self.step = int(step)
        super().__init__(conv, **kwargs)

    def begin(self):
        self.mobject.jump_to(self.step)
        super().begin()

    def interpolate_mobject(self, alpha):
        return self


class JumpWindow(Animation):
    def __init__(self, conv, step, **kwargs):
        self.step = int(step)
        super().__init__(conv, **kwargs)

    def begin(self):
        self.old_step = self.mobject.current_step
        self.mobject.kernel_row.drop_labels()
        super().begin()

    def interpolate_mobject(self, alpha):
        alpha = self.rate_func(alpha)
        if alpha >= 1:
            self.mobject.jump_to(self.step)
        else:
            self.mobject.apply_kernel_fade(self.old_step, self.step, alpha) 


class Convolution1D(Group):
    def __init__(
        self,
        signal,
        kernel,
        zero_pad=True,
        show_labels=False,
        signal_positive_color=BLUE,
        signal_negative_color=RED,
        kernel_positive_color=RED,
        kernel_negative_color=BLUE,
        result_positive_color=BLUE,
        result_negative_color=RED,
        start=LEFT * 2.5,
        end=RIGHT * 2.5,
        spacing=0.35,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.signal_values = [float(value) for value in signal]
        self.weights = [float(value) for value in kernel]
        if not self.signal_values:
            raise ValueError("signal is empty")
        if not self.weights:
            raise ValueError("kernel is empty")
        self.kernel_width = len(self.weights)
        self.zero_pad = zero_pad
        self.show_labels = show_labels
        if not zero_pad and self.kernel_width > len(self.signal_values):
            raise ValueError("kernel is wider than the signal; turn zero_pad on or use a shorter kernel")

        self.pad_left = (self.kernel_width - 1) // 2 if zero_pad else 0
        self.pad_right = self.kernel_width // 2 if zero_pad else 0
        self.padded_signal = (
            [0.0] * self.pad_left + self.signal_values + [0.0] * self.pad_right
        )
        self.n_columns = len(self.padded_signal)
        self.n_steps = self.n_columns - self.kernel_width + 1
        self.dots = []
        self.result_columns = []
        for step in range(self.n_steps):
            window = self.padded_signal[step:step + self.kernel_width]
            self.dots.append(sum(sample * weight for sample, weight in zip(window, self.weights)))
            self.result_columns.append(step + self.kernel_width // 2)
        self.signal_peak = self._peak(self.signal_values)
        self.kernel_peak = self._peak(self.weights)
        self.result_peak = self._peak(self.dots)
        self.current_step = None
        self.revealed = set()
        self.row_start = start
        self.row_end = end
        self.spacing = spacing

        self.signal_row = self._make_row(
            signal_positive_color,
            signal_negative_color,
        )
        self.kernel_row = self._make_row(
            kernel_positive_color,
            kernel_negative_color,
        )
        self.result_row = self._make_row(
            result_positive_color,
            result_negative_color,
        )
        gap = 1.5 * self.signal_row.height
        self.kernel_row.shift(UP * gap)
        self.result_row.shift(UP * 2 * gap)
        self.add(self.signal_row, self.kernel_row, self.result_row)
        self._show_signal()

    def _make_row(self, positive_color, negative_color):
        row = RowOfCubes(
            [0] * self.n_columns,
            start=self.row_start,
            end=self.row_end,
            spacing=self.spacing,
            show_labels=False,
            fill_color=interpolate_color(negative_color, positive_color, 0.5),
            fill_opacity=0,
        )
        row.positive_color = positive_color
        row.negative_color = negative_color
        for index in range(self.n_columns):
            row.set_body_visible(index, False)
        return row

    def _show_signal(self):
        signal_end = self.pad_left + len(self.signal_values)
        for index, value in enumerate(self.padded_signal):
            if index < self.pad_left or index >= signal_end:
                continue
            self.signal_row.set_body_visible(
                index,
                True,
                fill_opacity=1,
                color=self._color_for_value(
                    value,
                    self.signal_peak,
                    self.signal_row,
                ),
            )
            if self.show_labels:
                self.signal_row._add_label(value, self.signal_row.cubes[index])

    @staticmethod
    def _peak(values):
        return max((abs(float(value)) for value in values), default=0.0)

    @staticmethod
    def _color_for_value(value, peak, row):
        normalized = 0 if peak <= 0 else np.clip(float(value) / peak, -1, 1)
        return interpolate_color(
            row.negative_color,
            row.positive_color,
            (normalized + 1) / 2,
        )

    @property
    def labels(self):
        return [
            *self.signal_row.labels,
            *self.kernel_row.labels,
            *self.result_row.labels,
        ]

    def kernel_window(self, step):
        if step is None:
            return []
        return list(range(step, step + self.kernel_width))

    def _kernel_value(self, index, step):
        if step is None:
            return None
        offset = index - step
        if offset < 0 or offset >= self.kernel_width:
            return None
        return self.weights[offset]

    def apply_kernel_fade(self, old_step, new_step, alpha):
        old = set(self.kernel_window(old_step))
        new = set(self.kernel_window(new_step))
        for index in range(self.n_columns):
            in_old = index in old
            in_new = index in new
            if not in_old and not in_new:
                self.kernel_row.set_cube_fill_opacity(index, 0, frame_opacity=0)
                continue
            old_value = self._kernel_value(index, old_step)
            new_value = self._kernel_value(index, new_step)
            if old_value is None:
                old_value = new_value
            if new_value is None:
                new_value = old_value
            color = interpolate_color(
                self._color_for_value(
                    old_value,
                    self.kernel_peak,
                    self.kernel_row,
                ),
                self._color_for_value(
                    new_value,
                    self.kernel_peak,
                    self.kernel_row,
                ),
                alpha,
            )
            frame = (1 - alpha) * float(in_old) + alpha * float(in_new)
            self.kernel_row.set_cube_fill_opacity(
                index,
                frame,
                frame_opacity=frame,
                color=color,
            )
        return self

    def _has_label(self, row, index):
        body = row.cubes[index]
        return any(label.tracked_body is body for label in row.labels)

    def jump_to(self, step):
        step = int(step)
        if step < 0 or step >= self.n_steps:
            raise IndexError(f"step {step} is outside 0..{self.n_steps - 1}")
        self.kernel_row.drop_labels()
        for index in range(self.n_columns):
            offset = index - step
            if 0 <= offset < self.kernel_width:
                weight = self.weights[offset]
                self.kernel_row.set_body_visible(
                    index,
                    True,
                    fill_opacity=1,
                    color=self._color_for_value(
                        weight,
                        self.kernel_peak,
                        self.kernel_row,
                    ),
                )
                if self.show_labels:
                    self.kernel_row._add_label(weight, self.kernel_row.cubes[index])
            else:
                self.kernel_row.set_body_visible(index, False)

        column = self.result_columns[step]
        dot = self.dots[step]
        self.result_row.set_body_visible(
            column,
            True,
            fill_opacity=1,
            color=self._color_for_value(
                dot,
                self.result_peak,
                self.result_row,
            ),
        )
        if self.show_labels and not self._has_label(self.result_row, column):
            self.result_row._add_label(dot, self.result_row.cubes[column])
        self.revealed.add(column)
        self.current_step = step
        return self

    def jump_animation(self, step, fade=False, **kwargs):
        if fade:
            return JumpWindow(self, step, **kwargs)
        return SnapJump(self, step, **kwargs)

    def fade_kernel(self, **kwargs):
        return AnimationGroup(
            self.kernel_row.animate.set_opacity(0),
            *(label.animate.set_opacity(0) for label in self.kernel_row.labels),
            **kwargs,
        )


class ConvolutionLayer(Convolution1D):
    """A convolution that uses another convolution's result row as its input."""

    def __init__(self, previous_convolution, kernel, zero_pad=None, **kwargs):
        if not isinstance(previous_convolution, Convolution1D):
            raise TypeError("previous_convolution must be a Convolution1D")

        self.previous_convolution = previous_convolution
        self.input_row = previous_convolution.result_row
        if zero_pad is None:
            zero_pad = previous_convolution.zero_pad

        super().__init__(
            previous_convolution.dots,
            kernel,
            zero_pad=zero_pad,
            show_labels=previous_convolution.show_labels,
            signal_positive_color=self.input_row.positive_color,
            signal_negative_color=self.input_row.negative_color,
            kernel_positive_color=previous_convolution.kernel_row.positive_color,
            kernel_negative_color=previous_convolution.kernel_row.negative_color,
            result_positive_color=previous_convolution.result_row.positive_color,
            result_negative_color=previous_convolution.result_row.negative_color,
            start=previous_convolution.row_start,
            end=previous_convolution.row_end,
            spacing=previous_convolution.spacing,
            **kwargs,
        )

        # The prior result row is the input display for this layer, so do not
        # render the temporary signal row created by Convolution1D.
        self.signal_row.drop_labels()
        self.shift(
            self.input_row.get_center() - self.signal_row.get_center(),
        )
        self.remove(self.signal_row)


class Main(ThreeDScene):
    def interactive(self, dt=1 / 30):
        while self.renderer.window is not None and not self.renderer.window.is_closing:
            self.update_mobjects(dt)
            self.renderer.render(self, dt, [])

    def construct(self):
        conv = Convolution1D(
            [0, 0, 2, 2, 2],
            [-1, 0, 1],
            zero_pad=True,
            show_labels=False,
            signal_positive_color=RED,
            signal_negative_color=BLUE,
            kernel_positive_color=RED,
            kernel_negative_color=BLUE,
            result_positive_color=RED,
            result_negative_color=BLUE,
            start=LEFT * 2.5,
            end=RIGHT * 2.5,
            spacing=0.35,
        )
        conv.to_edge(DOWN) 
        self.add(conv)

        conv2 = ConvolutionLayer(conv, [-1, 1, -1])
        self.add(conv2)

        if config.renderer == RendererType.CAIRO:
            self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
            self.camera.add_fixed_orientation_mobjects(*conv.labels)

            for step in range(conv.n_steps):
                self.play(conv.jump_animation(step, fade=True), run_time=0.5)
                self.wait(0.5)
                self.play(conv.fade_kernel(), run_time=0.25)
        else:
            for step in range(conv.n_steps):
                self.play(conv.jump_animation(step), run_time=0.5)

            for step in range(conv2.n_steps):
                self.play(conv2.jump_animation(step), run_time=0.5)
            self.wait(0.5)
            self.interactive()
