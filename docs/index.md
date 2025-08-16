# Lithic Editor and Annotator

<div class="hero-section">
  <h2>Lithic Editor and Annotator</h2>
  <p>Automatically remove ripple lines from lithic drawings, annotate illustrations with directional arrows, and produce publication ready drawings</p>
</div>

!!! success "Key Features"
    - ![](assets/images/lithic_tool.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px"}**Intelligent Ripple Removal** - Advanced algorithms distinguish between structural elements and scar ripples
    - ![](assets/images/arrow.svg){: style="width:24px; height:24px; transform:rotate(-45deg); vertical-align:middle; margin-right:8px; filter:brightness(0)"}**Precise Annotations** - Add directional arrows to indicate striking patterns
    - ![](assets/images/article.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px; filter:brightness(0)"}**Publication Ready** - Maintain DPI and produce high-quality outputs
    - ![](assets/images/smile_face.svg){: style="width:24px; height:24px; vertical-align:middle; margin-right:8px; filter:brightness(0)"}**Easy to Use** - Intuitive GUI and command-line interface

## What is Lithic Editor?

Lithic Editor and Annotator is a specialized tool designed for archaeological research, specifically for processing technical drawings of lithic artifacts. It addresses two critical challenges in lithic illustration:

1. **Automated Ripple Line Removal**: The software uses advanced graph-based algorithms to identify and remove hatching/ripple lines while preserving the essential structural elements of the drawing.

2. **Technical Annotation System**: Provides intuitive tools for replacing scar ripples with directional arrows to indicate striking direction and flake scar patterns.

## Quick Start

=== "GUI Application"

    ```bash
    # Install the package
    pip install git+https://github.com/JasonGellis/lithic-editor.git

    # Launch the GUI
    lithic-editor --gui
    ```

=== "Python API"

    ```python
    from lithic_editor.processing import process_lithic_drawing

    # Process an image
    result = process_lithic_drawing(
        "lithic_drawing.png",
        save_debug=True
    )
    ```

=== "Command Line"

    ```bash
    # Process a single image
    lithic-editor process drawing.png --output results/
    ```

## Visual Example

<div class="comparison-container">
  <div class="before-after">
    <div class="image-box">
      <h3>Before Processing</h3>
      <p>Original drawing with ripple lines</p>
      <!-- Add your before image here -->
    </div>
    <div class="image-box">
      <h3>After Processing</h3>
      <p>Clean structural elements preserved</p>
      <!-- Add your after image here -->
    </div>
  </div>
</div>

## Who is this for?

- **Archaeologists** working with lithic illustrations
- **Researchers** creating publication ready images
- **Museum Curators** preparing artifact documentation
- **Students** studying archaeological illustration techniques

## Getting Started

<div class="card-grid">
  <div class="card">
    <h3>Installation</h3>
    <p>Set up Lithic Editor on your system</p>
    <a href="getting-started/installation/" class="md-button">Install Guide</a>
  </div>

  <div class="card">
    <h3>Quick Start</h3>
    <p>Process your first lithic drawing</p>
    <a href="getting-started/quickstart/" class="md-button">Tutorial</a>
  </div>

  <div class="card">
    <h3>User Guide</h3>
    <p>Learn all features and workflows</p>
    <a href="user-guide/overview/" class="md-button">Documentation</a>
  </div>
</div>

## Support

- **Issues**: [GitHub Issues](https://github.com/JasonGellis/lithic-editor/issues)

---

!!! info "Acknowledgements"
    Thank you to The British Academy for funding this project.