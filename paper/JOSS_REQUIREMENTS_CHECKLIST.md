# JOSS Submission Requirements Checklist

This document outlines what needs to be completed to submit the Lithic Editor and Annotator to the Journal of Open Source Software (JOSS).

## ✅ Already Complete

### Software Requirements
- [x] **Open source license**: MIT License is in place
- [x] **Substantial scholarly effort**: 3+ months of development evident
- [x] **Feature-complete**: Core functionality implemented including cortex preservation and DPI upscaling
- [x] **300+ lines of code**: Package contains ~3,000+ lines with new features
- [x] **Research application**: Clear archaeological research use case with cortex documentation
- [x] **Package structure**: Professional Python package with modular upscaling/cortex components

### Basic Documentation
- [x] **README.md**: Comprehensive documentation with installation and usage
- [x] **API documentation**: Docstrings throughout codebase (PEP8 compliant)
- [x] **CHANGELOG.md**: Version history and semantic versioning
- [x] **Test suite**: pytest-based testing framework with cortex/upscaling coverage
- [x] **Feature documentation**: MkDocs site with cortex preservation and upscaling sections
- [x] **CLI help**: Updated with --no-preserve-cortex parameter

### Paper Draft
- [x] **Paper structure**: Following JOSS format requirements
- [x] **Bibliography**: Academic references in .bib format (including ESPCN/FSRCNN citations)
- [x] **Statement of need**: Clear problem identification
- [x] **Summary**: Concise software description with cortex and upscaling features
- [x] **Feature documentation**: Updated to include DPI-aware upscaling and cortex preservation

## ❌ Missing/Needs Completion

### Critical Requirements for JOSS Submission

#### 1. Author Information and ORCIDs
- [x] **Complete author details**: Jason Jacob Gellis information added
- [x] **ORCID IDs**: Valid ORCID identifier (0000-0002-9929-789X) added
- [x] **Institutional affiliations**: University of Cambridge & British Academy affiliations complete
- [x] **Corresponding author**: Contact information verified (jg760@cam.ac.uk)

#### 2. Real Academic References
- [ ] **Replace placeholder citations**: Current bibliography contains fictional references
- [ ] **Archaeological literature review**: Add relevant lithic analysis publications
- [ ] **Image processing citations**: Include real computer vision/image processing papers
- [ ] **Software citations**: Reference actual related software packages

#### 3. Validation and Performance Data
- [ ] **Empirical validation**: Need real testing data on archaeological illustrations
- [ ] **Performance metrics**: Actual accuracy measurements vs. manual processing
- [ ] **Comparison studies**: Quantitative comparison with existing tools
- [ ] **User studies**: Evidence of adoption and community feedback

#### 4. Example Data and Figures
- [ ] **Figure 1**: Example input/output showing ripple removal
- [ ] **Figure 2**: Arrow annotation demonstration
- [ ] **Figure 3**: Performance comparison charts
- [ ] **Test dataset**: Representative archaeological illustrations for validation

#### 5. Documentation Enhancements
- [ ] **Installation verification**: Test installation instructions on clean systems
- [ ] **Tutorial notebooks**: Jupyter notebooks demonstrating key workflows
- [ ] **API reference**: Complete function documentation
- [ ] **Troubleshooting guide**: Common issues and solutions

#### 6. Community and Reproducibility
- [ ] **Issue tracker**: GitHub issues enabled and responsive
- [ ] **Community guidelines**: CONTRIBUTING.md file
- [ ] **Code of conduct**: Community interaction guidelines
- [ ] **Reproducible examples**: Datasets and scripts for paper results

#### 7. Software Quality Assurance
- [ ] **Continuous integration**: GitHub Actions or similar CI/CD
- [x] **Test coverage**: Comprehensive test suite including upscaling, cortex, GUI, CLI, and integration tests
- [ ] **Code quality badges**: Add status badges to README
- [ ] **Linting compliance**: Ensure flake8/black compliance

### Repository Setup

#### 8. GitHub Repository Preparation
- [ ] **Public repository**: Ensure repository is publicly accessible
- [ ] **Release tags**: Tagged releases for version management
- [ ] **DOI assignment**: Consider Zenodo integration for citable releases
- [ ] **License file**: Ensure LICENSE file is present and correct

#### 9. Archive and Preservation
- [ ] **Zenodo archival**: Create DOI for software citation
- [ ] **Long-term preservation**: Ensure institutional backup
- [ ] **Data preservation**: Archive test datasets appropriately

## Recommended Timeline for Completion

### Phase 1 (Week 1-2): Core Requirements
1. Gather real author information and ORCID IDs
2. Replace fictional references with real academic citations
3. Create authentic test dataset with archaeological illustrations
4. Generate real performance validation data

### Phase 2 (Week 3-4): Documentation and Quality
1. Add example figures and illustrations
2. Create tutorial notebooks
3. Implement continuous integration
4. Improve test coverage

### Phase 3 (Week 5-6): Community Preparation
1. Setup issue tracking and community guidelines
2. Create Zenodo DOI
3. Final paper revision and peer review
4. Submit to JOSS

## Key Resources Needed

### Academic Collaboration
- [ ] **Archaeological collaborators**: Researchers with lithic illustration datasets
- [ ] **Validation datasets**: Access to published lithic drawings for testing
- [ ] **Domain experts**: Archaeologists for validation and feedback

### Technical Resources
- [ ] **CI/CD setup**: GitHub Actions configuration
- [ ] **Test infrastructure**: Automated testing across platforms
- [ ] **Documentation hosting**: GitHub Pages or ReadTheDocs

### Publication Support
- [ ] **Institutional support**: Verify institutional backing for publication
- [ ] **Funding acknowledgment**: Complete funding source information
- [ ] **Ethics clearance**: Ensure any archaeological data use is properly cleared

## Success Criteria

The submission will be ready when:

1. **All authors have complete ORCID profiles** and institutional affiliations
2. **Real validation data** demonstrates software effectiveness on archaeological materials
3. **Complete academic bibliography** with relevant domain literature
4. **Reproducible examples** allow reviewers to verify claims
5. **Professional documentation** meets academic software standards
6. **Community infrastructure** supports ongoing development and user support

## Notes for Implementation

- **Priority**: Focus on real validation data and academic references first
- **Community**: Engage archaeological community early for feedback and validation
- **Quality**: Maintain high software quality standards throughout development
- **Timeline**: Allow 6-8 weeks for complete preparation before JOSS submission