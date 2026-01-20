// ============================= INITIALIZATION ============================= //
document.addEventListener('DOMContentLoaded', () => {
    // Parse plot data from the embedded JSON (legacy support)
    const plotDataElement = document.getElementById('plot-data');
    window.plotData = plotDataElement ? JSON.parse(plotDataElement.textContent) : {};
    
    // Initialize legacy dropdowns (backward compatibility)
    document.querySelectorAll('.figure-dropdown:not(.plotly-selector-container .figure-dropdown)').forEach(dropdown => {
        const firstOption = dropdown.options[0];
        if (firstOption) {
            showFigure(dropdown, dropdown.dataset.containerId || dropdown.closest('.figure-container').id);
        }
    });
    
    // Initialize all dynamic tables with improved performance
    initializeTables();
    
    // Initialize Plotly selectors
    initializePlotlySelectors();
});

// ========================== PLOTLY SELECTOR INITIALIZATION ========================== //
function initializePlotlySelectors() {
    document.querySelectorAll('.plotly-selector-container').forEach(container => {
        const selector = container.querySelector('.figure-dropdown');
        const plotDiv = container.querySelector('.plotly-selector-plot');
        
        if (selector && selector.options.length > 0) {
            // Trigger initial plot display
            const firstOption = selector.options[0];
            if (firstOption) {
                selector.value = firstOption.value;
                // The plot will be displayed by the embedded script in each selector
            }
        }
        
        // Fix Plotly container styling after initialization
        fixPlotlyContainers();
    });
}

// ============================ FIX PLOTLY CONTAINER STYLING ============================ //
function fixPlotlyContainers() {
    // Wait for Plotly to initialize, then fix container styling
    setTimeout(() => {
        document.querySelectorAll('.plotly-selector-plot .svg-container').forEach(container => {
            // Remove problematic inline styles that cause layout issues
            container.style.height = 'auto';
            container.style.minHeight = '400px'; // Set a reasonable minimum height
        });
    }, 100);
}

// ============================ IMPROVED TABLE INITIALIZATION ============================ //
function initializeTables() {
    const tables = document.querySelectorAll('.dynamic-table');
    
    // Use requestAnimationFrame to prevent blocking
    const initTable = (index) => {
        if (index >= tables.length) return;
        
        const table = tables[index];
        makeTableSortable(table);
        makeTableResizable(table);
        
        // Initialize pagination for each table
        const tableId = table.id;
        if (tableId) {
            const container = table.closest('.table-container');
            const select = container?.querySelector('.rows-per-page');
            if (select) {
                updatePagination(tableId, parseInt(select.value, 10), 0);
            }
        }
        
        // Process next table asynchronously
        requestAnimationFrame(() => initTable(index + 1));
    };
    
    initTable(0);
}

// ============================ LEGACY PLOT RENDERING ============================ //
function showFigure(dropdown, containerId) {
    const plotId = dropdown.value;
    const container = document.getElementById(containerId);
    
    if (!plotId || !container) return;
    
    // Clear previous content
    container.innerHTML = `<div id="container-${plotId}" class="plot-container"></div>`;
    
    // Initialize the plot
    initializePlot(plotId);
}

function initializePlot(plotId) {
    const plotInfo = window.plotData[plotId];
    const container = document.getElementById(`container-${plotId}`);
    if (!plotInfo || !container) {
        console.error(`Data or container not found for plot: ${plotId}`);
        return;
    }
    
    // Clear previous content and skip if already plotted
    if (container.classList.contains('js-plotly-plot') || container.querySelector('img')) {
        return;
    }
    container.innerHTML = '';

    switch(plotInfo.type) {
        case 'plotly':
            Plotly.newPlot(container, plotInfo.data, plotInfo.layout, { 
                responsive: true,
                displayModeBar: true
            }).then(() => {
                // Fix Plotly container styling after plot is rendered
                fixPlotlyContainers();
            });
            break;
        case 'image':
            const img = document.createElement('img');
            img.src = `data:image/png;base64,${plotInfo.data}`;
            img.style.maxWidth = '100%';
            img.style.height = 'auto';
            img.alt = 'Generated plot';
            container.appendChild(img);
            break;
        case 'error':
            container.innerHTML = `<div class="error-message">Error: ${plotInfo.error}</div>`;
            break;
        default:
            container.innerHTML = `<div class="error-message">Unsupported plot type: ${plotInfo.type}</div>`;
    }
}

// ========================== SECTION COLLAPSE/EXPAND ========================= //
function toggleSection(event) {
    const header = event.currentTarget;
    const section = header.closest('.section');
    section.classList.toggle('collapsed');
    
    // Trigger resize event for any Plotly plots in the section
    setTimeout(() => {
        const plotlyDivs = section.querySelectorAll('.js-plotly-plot');
        plotlyDivs.forEach(div => {
            if (window.Plotly) {
                Plotly.Plots.resize(div);
                // Fix container styling after resize
                setTimeout(fixPlotlyContainers, 50);
            }
        });
    }, 400); // Wait for CSS transition to complete
}

function toggleAllSections(expand) {
    document.querySelectorAll('.section').forEach(section => {
        if (expand) {
            section.classList.remove('collapsed');
        } else {
            section.classList.add('collapsed');
        }
    });
    
    // Trigger resize for visible plots after expansion
    if (expand) {
        setTimeout(() => {
            document.querySelectorAll('.js-plotly-plot').forEach(div => {
                if (window.Plotly) {
                    Plotly.Plots.resize(div);
                    // Fix container styling after resize
                    setTimeout(fixPlotlyContainers, 50);
                }
            });
        }, 500);
    }
}

// ============================== IMPROVED TABLE SORTING =============================== //
function makeTableSortable(table) {
    const headers = table.querySelectorAll('th');
    
    headers.forEach((header, index) => {
        // Skip if header already has click listener
        if (header.dataset.sortable === 'true') return;
        header.dataset.sortable = 'true';
        
        header.addEventListener('click', (e) => {
            // Don't sort if clicking on resize handle
            if (e.target.classList.contains('resizable-handle')) return;
            
            const currentSort = header.dataset.sortDirection || 'none';
            const newSort = currentSort === 'asc' ? 'desc' : 'asc';
            
            sortTableByColumn(table, index, newSort);
            
            // Update header indicators
            headers.forEach(h => {
                h.classList.remove('asc', 'desc');
                h.dataset.sortDirection = 'none';
            });
            
            header.classList.add(newSort);
            header.dataset.sortDirection = newSort;
            
            // Reset pagination to first page after sorting
            const container = table.closest('.table-container');
            const select = container?.querySelector('.rows-per-page');
            if (select && table.id) {
                updatePagination(table.id, parseInt(select.value), 0);
            }
        });
    });
}

function sortTableByColumn(table, columnIndex, direction) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const sortedRows = rows.sort((a, b) => {
        const aCell = a.cells[columnIndex];
        const bCell = b.cells[columnIndex];
        
        if (!aCell || !bCell) return 0;
        
        const aText = aCell.textContent.trim();
        const bText = bCell.textContent.trim();
        
        // Try numeric comparison first
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
            return direction === 'asc' ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string comparison
        const result = aText.localeCompare(bText, undefined, { 
            numeric: true, 
            sensitivity: 'base' 
        });
        
        return direction === 'asc' ? result : -result;
    });
    
    // Use DocumentFragment for efficient DOM manipulation
    const fragment = document.createDocumentFragment();
    sortedRows.forEach(row => fragment.appendChild(row));
    tbody.appendChild(fragment);
}

// ============================ IMPROVED TABLE COLUMN RESIZING =========================== //
function makeTableResizable(table) {
    const headers = table.querySelectorAll('th');
    
    headers.forEach(header => {
        // Skip if already has resize handle
        if (header.querySelector('.resizable-handle')) return;
        
        const handle = document.createElement('div');
        handle.className = 'resizable-handle';
        header.appendChild(handle);

        let startX, startWidth, isResizing = false;

        const onMouseMove = (e) => {
            if (!isResizing) return;
            
            const newWidth = Math.max(50, startWidth + (e.clientX - startX));
            header.style.width = `${newWidth}px`;
            
            // Prevent text selection during resize
            e.preventDefault();
        };

        const onMouseUp = () => {
            if (!isResizing) return;
            
            isResizing = false;
            document.removeEventListener('mousemove', onMouseMove);
            document.removeEventListener('mouseup', onMouseUp);
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
        };

        handle.addEventListener('mousedown', (e) => {
            e.stopPropagation(); // Prevent sorting
            e.preventDefault();
            
            isResizing = true;
            startX = e.clientX;
            startWidth = header.offsetWidth;
            
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
            
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    });
}

// ============================ IMPROVED TABLE PAGINATION ============================== //
function changePageSize(tableId, size) {
    const sizeInt = parseInt(size, 10);
    updatePagination(tableId, sizeInt, 0);
    
    // Store preference in localStorage if available
    try {
        localStorage.setItem(`pageSize_${tableId}`, size);
    } catch (e) {
        // Ignore localStorage errors
    }
}

function goToPage(tableId, pageSize, pageIndex) {
    updatePagination(tableId, pageSize, pageIndex);
}

function updatePagination(tableId, pageSize, pageIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;

    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const totalRows = rows.length;
    const isPaginated = pageSize > 0 && totalRows > pageSize;
    
    const totalPages = isPaginated ? Math.ceil(totalRows / pageSize) : 1;
    const currentPage = Math.min(Math.max(0, pageIndex), totalPages - 1);

    const start = isPaginated ? currentPage * pageSize : 0;
    const end = isPaginated ? Math.min(start + pageSize, totalRows) : totalRows;

    // Show/hide rows efficiently
    rows.forEach((row, i) => {
        row.style.display = (i >= start && i < end) ? '' : 'none';
    });

    // Update controls
    updatePaginationControls(tableId, totalPages, currentPage, totalRows, start, end);
}

function updatePaginationControls(tableId, totalPages, currentPage, totalRows, start, end) {
    // Escape special characters in tableId
    const escapedTableId = CSS.escape(tableId);
    
    const container = document.getElementById(escapedTableId)?.closest('.table-container');
    if (!container) return;
    
    const paginationContainer = container.querySelector(`#pagination-${CSS.escape(tableId)}`);
    const indicator = container.querySelector(`#indicator-${CSS.escape(tableId)}`);
    
    if (!paginationContainer || !indicator) return;

    // Clear previous buttons
    paginationContainer.innerHTML = '';
    
    if (totalPages > 1) {
        // Create pagination buttons with smart truncation
        createPaginationButtons(paginationContainer, tableId, totalPages, currentPage);
        indicator.textContent = `Showing ${start + 1}-${end} of ${totalRows} rows`;
    } else {
        indicator.textContent = `Showing all ${totalRows} rows`;
    }
}

function createPaginationButtons(container, tableId, totalPages, currentPage) {
    const maxVisibleButtons = 7;
    
    // Previous button
    if (currentPage > 0) {
        const prevBtn = createPaginationButton('‹', () => goToPage(tableId, getPageSize(tableId), currentPage - 1));
        prevBtn.title = 'Previous page';
        container.appendChild(prevBtn);
    }
    
    // Calculate visible page range
    let startPage = Math.max(0, currentPage - Math.floor(maxVisibleButtons / 2));
    let endPage = Math.min(totalPages - 1, startPage + maxVisibleButtons - 1);
    
    // Adjust start if we're near the end
    if (endPage - startPage < maxVisibleButtons - 1) {
        startPage = Math.max(0, endPage - maxVisibleButtons + 1);
    }
    
    // First page + ellipsis
    if (startPage > 0) {
        container.appendChild(createPaginationButton(1, () => goToPage(tableId, getPageSize(tableId), 0)));
        if (startPage > 1) {
            const ellipsis = document.createElement('span');
            ellipsis.textContent = '...';
            ellipsis.className = 'pagination-ellipsis';
            container.appendChild(ellipsis);
        }
    }
    
    // Page buttons
    for (let i = startPage; i <= endPage; i++) {
        const btn = createPaginationButton(i + 1, () => goToPage(tableId, getPageSize(tableId), i));
        if (i === currentPage) {
            btn.classList.add('active');
        }
        container.appendChild(btn);
    }
    
    // Ellipsis + last page
    if (endPage < totalPages - 1) {
        if (endPage < totalPages - 2) {
            const ellipsis = document.createElement('span');
            ellipsis.textContent = '...';
            ellipsis.className = 'pagination-ellipsis';
            container.appendChild(ellipsis);
        }
        container.appendChild(createPaginationButton(totalPages, () => goToPage(tableId, getPageSize(tableId), totalPages - 1)));
    }
    
    // Next button
    if (currentPage < totalPages - 1) {
        const nextBtn = createPaginationButton('›', () => goToPage(tableId, getPageSize(tableId), currentPage + 1));
        nextBtn.title = 'Next page';
        container.appendChild(nextBtn);
    }
}

function createPaginationButton(text, onClick) {
    const btn = document.createElement('button');
    btn.textContent = text;
    btn.className = 'pagination-btn';
    btn.onclick = onClick;
    return btn;
}

function getPageSize(tableId) {
    const container = document.getElementById(tableId)?.closest('.table-container');
    const select = container?.querySelector('.rows-per-page');
    return select ? parseInt(select.value, 10) : 10;
}

// ============================ UTILITY FUNCTIONS ============================== //

// Debounce function for performance optimization
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Handle window resize for responsive tables
window.addEventListener('resize', debounce(() => {
    // Resize Plotly plots
    document.querySelectorAll('.js-plotly-plot').forEach(div => {
        if (window.Plotly) {
            Plotly.Plots.resize(div);
            // Fix container styling after resize
            setTimeout(fixPlotlyContainers, 50);
        }
    });
}, 250));

// Export functions for external access
window.TableUtils = {
    toggleSection,
    toggleAllSections,
    changePageSize,
    goToPage,
    showFigure,
    initializePlot,
    fixPlotlyContainers
};
