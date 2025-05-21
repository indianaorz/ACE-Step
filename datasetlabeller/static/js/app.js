document.addEventListener('DOMContentLoaded', function () {
    const trackListElement = document.getElementById('trackList');
    const editorPanel = document.getElementById('editorPanel');
    const editingTrackKeySpan = document.querySelector('#editingTrackKey span');
    const audioPlayer = document.getElementById('audioPlayer');
    
    const gameInput = document.getElementById('game');
    const bossInput = document.getElementById('boss');
    const stageInput = document.getElementById('stage');
    const descriptionInput = document.getElementById('description');
    const currentTrackKeyInput = document.getElementById('currentTrackKey');
    const saveButton = document.getElementById('saveButton');
    const saveStatusElement = document.getElementById('saveStatus');

    const tagsContainer = document.getElementById('tagsContainer');
    const tagInput = document.getElementById('tagInput');
    const instrumentsContainer = document.getElementById('instrumentsContainer');
    const instrumentInput = document.getElementById('instrumentInput');
    const trackSearchInput = document.getElementById('trackSearchInput');

    const bpmInput      = document.getElementById('bpm');
    const detectBpmButton = document.getElementById('detectBpmButton');
    const musicKeyInput = document.getElementById('musicKey');
    const lyricsInput    = document.getElementById('lyrics'); 


    let currentSelectedListItem = null;

    // --- Track List Interaction ---
    trackListElement.addEventListener('click', async function (e) {
        let targetLink = e.target.closest('a');
        if (!targetLink) return;
        e.preventDefault();

        const listItem = targetLink.parentElement;
        const trackKey = listItem.dataset.trackKey;

        if (currentSelectedListItem) {
            currentSelectedListItem.classList.remove('active');
        }
        listItem.classList.add('active');
        currentSelectedListItem = listItem;

        loadTrackForEditing(trackKey);
    });

    async function loadTrackForEditing(trackKey) {
        saveStatusElement.textContent = 'Loading track...';
        saveStatusElement.className = 'status-message'; 
        try {
            const response = await fetch(`/get_track_details/${encodeURIComponent(trackKey)}`);
            if (!response.ok) {
                const errorResult = await response.json().catch(() => ({ message: `HTTP error! Status: ${response.status}` }));
                throw new Error(errorResult.message || `HTTP error! Status: ${response.status}`);
            }
            const result = await response.json();

            if (result.status === 'success') {
                const trackData = result.data;
                currentTrackKeyInput.value = result.key;
                editingTrackKeySpan.textContent = result.key;
                audioPlayer.src = result.audio_url;

                gameInput.value = trackData.game || '';
                bossInput.value = trackData.boss || '';
                stageInput.value = trackData.stage || '';
                descriptionInput.value = trackData.description || '';
                lyricsInput.value      = trackData.lyrics  || '';  // <-- populate

                bpmInput.value       = trackData.bpm ?? '';
                musicKeyInput.value  = trackData.key ?? '';


                setupChipSystem(Array.isArray(trackData.tags) ? trackData.tags : [], tagsContainer, tagInput);
                setupChipSystem(Array.isArray(trackData.instruments) ? trackData.instruments : [], instrumentsContainer, instrumentInput);

                editorPanel.style.display = 'block';
                saveStatusElement.textContent = 'Track loaded.';
                saveStatusElement.classList.add('success');
                if (window.innerWidth < 900) {
                    editorPanel.scrollIntoView({ behavior: 'smooth' });
                }
            } else {
                throw new Error(result.message || 'Failed to load track data.');
            }
        } catch (error) {
            console.error('Failed to fetch track details:', error);
            saveStatusElement.textContent = `Error: ${error.message}`;
            saveStatusElement.classList.add('error');
            editorPanel.style.display = 'none';
        }
    }
    
    // --- Chip System (for Tags and Instruments) with Drag-and-Drop ---
    let draggedChipElement = null; 

    function setupChipSystem(initialItemsArray, container, inputElement) {
        let currentItems = (Array.isArray(initialItemsArray) ? initialItemsArray : [])
                            .map(item => typeof item === 'string' ? item.trim() : '')
                            .filter(Boolean);

        function renderChips() {
            container.innerHTML = ''; 
            currentItems.forEach(itemText => {
                const chip = document.createElement('span');
                chip.classList.add('chip');
                chip.textContent = itemText; 
                chip.draggable = true; 

                chip.addEventListener('dragstart', handleDragStart);
                chip.addEventListener('dragend', handleDragEnd);

                const removeBtn = document.createElement('span');
                removeBtn.classList.add('remove-chip');
                removeBtn.innerHTML = '&times;'; 
                removeBtn.title = `Remove ${itemText}`;
                removeBtn.onclick = (e) => {
                    e.stopPropagation(); 
                    currentItems = currentItems.filter(item => item !== itemText); 
                    renderChips(); 
                };
                chip.appendChild(removeBtn); 
                container.appendChild(chip);
            });
        }

        inputElement.onkeypress = function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                const newItem = inputElement.value.trim();
                if (newItem && !currentItems.includes(newItem)) { 
                    currentItems.push(newItem); 
                    renderChips(); 
                }
                inputElement.value = ''; 
            }
        };

        container.addEventListener('dragover', handleDragOver);
        container.addEventListener('drop', (e) => handleDropOnContainer(e, currentItems, renderChips, container));
        container.addEventListener('dragleave', (e) => { // Clean up if mouse leaves container during drag
            const relatedTargetIsWithin = e.relatedTarget && e.currentTarget.contains(e.relatedTarget);
            if (!relatedTargetIsWithin) {
                 e.currentTarget.classList.remove('drag-over');
            }
        });

        renderChips(); 
    }
    
    // Drag and Drop Handlers
    function handleDragStart(e) {
        if (!e.target.classList.contains('chip')) return; 
        draggedChipElement = e.target; 
        e.dataTransfer.effectAllowed = 'move';
        const itemText = draggedChipElement.childNodes[0].nodeValue.trim(); 
        e.dataTransfer.setData('text/plain', itemText); 
        
        setTimeout(() => { 
            if(draggedChipElement) draggedChipElement.classList.add('dragging');
        }, 0);
    }

    function handleDragEnd(e) {
        // This mainly handles cleanup if the drag was cancelled or dropped outside a valid target.
        // If dropped on a valid target, renderChips() in handleDropOnContainer will do the primary cleanup.
        if (draggedChipElement) {
            draggedChipElement.classList.remove('dragging');
        }
        draggedChipElement = null; 
        document.querySelectorAll('.chips-container.drag-over').forEach(cont => {
            cont.classList.remove('drag-over');
        });
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move'; // Sets the cursor icon

        const currentContainer = e.target.closest('.chips-container');

        // Initial guard clauses:
        // 1. Ensure we're over a valid chips container.
        // 2. Ensure there's an active dragged chip.
        // 3. Ensure the dragged chip actually belongs to (or is currently in) this container.
        //    draggedChipElement.parentNode === currentContainer check is good for reordering within the same container.
        if (!currentContainer || !draggedChipElement || draggedChipElement.parentNode !== currentContainer) {
            // If the chip is not from this container, or other essential conditions fail, do nothing.
            // This also handles the case where currentContainer might be null if not hovering over a valid one.
            return;
        }

        const afterElement = getDragAfterElement(currentContainer, e.clientY);

        // Now, perform the DOM manipulation for live reordering:
        if (afterElement === null) {
            // Intention: move draggedChipElement to the end of currentContainer.
            // Only do this if it's not already the last child, to prevent redundant DOM operations.
            if (currentContainer.lastChild !== draggedChipElement) {
                currentContainer.appendChild(draggedChipElement);
            }
        } else {
            // Intention: move draggedChipElement to be before afterElement.
            // Ensure we're not trying to insert the element before itself (shouldn't happen if afterElement is from filtered list).
            // Also, only do this if draggedChipElement is not already in that position.
            if (draggedChipElement !== afterElement && draggedChipElement.nextSibling !== afterElement) {
                currentContainer.insertBefore(draggedChipElement, afterElement);
            }
        }
    }

    function handleDropOnContainer(e, currentItemsArray, renderChipsCallback, containerElement) {
        e.preventDefault();
        containerElement.classList.remove('drag-over');

        if (!draggedChipElement || draggedChipElement.parentNode !== containerElement) {
            // Drop happened, but the dragged element is not valid or not from this container.
            // handleDragEnd will clean up draggedChipElement's class if it exists.
            return;
        }
        
        // The DOM might already be visually correct due to handleDragOver.
        // We now finalize by updating the underlying JavaScript array and re-rendering.

        const newOrderedItems = [];
        containerElement.querySelectorAll('.chip').forEach(chipEl => {
            if (chipEl.childNodes[0] && chipEl.childNodes[0].nodeValue) {
                 newOrderedItems.push(chipEl.childNodes[0].nodeValue.trim());
            }
        });
        
        currentItemsArray.length = 0; 
        newOrderedItems.forEach(item => currentItemsArray.push(item));
        
        // draggedChipElement is reset here, and renderChipsCallback will create fresh elements
        // without the .dragging class.
        draggedChipElement = null; 

        renderChipsCallback(); // This is crucial for final state and cleanup
    }

    // Use the simpler getDragAfterElement version
    function getDragAfterElement(container, y) {
        const draggableElements = Array.from(container.children)
            .filter(child => child.classList.contains('chip') && !child.classList.contains('dragging'));

        for (const child of draggableElements) {
            const box = child.getBoundingClientRect();
            if (y < box.top + box.height / 2) {
                return child;
            }
        }
        return null; 
    }

    function getChipsFromContainer(container) {
        const chips = [];
        container.querySelectorAll('.chip').forEach(chipElement => {
            if (chipElement.childNodes[0] && chipElement.childNodes[0].nodeValue) {
                chips.push(chipElement.childNodes[0].nodeValue.trim());
            }
        });
        return chips;
    }

    // --- Save Button ---
    saveButton.addEventListener('click', async function () {
        if (!currentTrackKeyInput.value) {
            saveStatusElement.textContent = 'No track selected to save.';
            saveStatusElement.className = 'status-message error';
            return;
        }

        const trackDataToSave = {
            game:        gameInput.value.trim(),
            boss:        bossInput.value.trim() || null,
            stage:       stageInput.value.trim(),
            description: descriptionInput.value.trim(),
            lyrics:      lyricsInput.value.trim(),
            bpm:         bpmInput.value ? Number(bpmInput.value) : null,
            key:         musicKeyInput.value.trim(),
            tags:        getChipsFromContainer(tagsContainer),
            instruments: getChipsFromContainer(instrumentsContainer)
          };
          

        saveStatusElement.textContent = 'Saving...';
        saveStatusElement.className = 'status-message';

        try {
            const response = await fetch('/save_track_details', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    key: currentTrackKeyInput.value,
                    data: trackDataToSave
                }),
            });

            const result = await response.json();
            if (response.ok && result.status === 'success') {
                saveStatusElement.textContent = result.message || 'Saved successfully!';
                saveStatusElement.classList.add('success');
                // setTimeout(() => { 
                //     saveStatusElement.textContent = ''; 
                //     window.location.reload(); 
                // }, 1500);
            } else {
                throw new Error(result.message || 'Failed to save data.');
            }
        } catch (error) {
            console.error('Failed to save track details:', error);
            saveStatusElement.textContent = `Error: ${error.message}`;
            saveStatusElement.classList.add('error');
        }
    });

    // --- Track Search/Filter ---
    trackSearchInput.addEventListener('input', function() {
        const searchTerm = trackSearchInput.value.toLowerCase();
        const tracks = trackListElement.getElementsByTagName('li');
        for (let i = 0; i < tracks.length; i++) {
            const trackKey = tracks[i].dataset.trackKey;
            if (trackKey && trackKey.toLowerCase().includes(searchTerm)) {
                tracks[i].style.display = '';
            } else {
                tracks[i].style.display = 'none';
            }
        }
    });


    // ——— BPM Detection ———
    detectBpmButton.addEventListener('click', async () => {
       if (!currentTrackKeyInput.value) return;
        saveStatusElement.textContent = 'Detecting BPM…';
        saveStatusElement.className = 'status-message';
    
        try {
            const resp = await fetch(
                `/detect_bpm/${encodeURIComponent(currentTrackKeyInput.value)}`
            );
            const result = await resp.json();
            if (resp.ok && result.status === 'success') {
                bpmInput.value = Math.round(result.bpm);
                saveStatusElement.textContent = `Detected BPM: ${Math.round(result.bpm)}`;
                saveStatusElement.classList.add('success');
            } else {
                throw new Error(result.message || 'Detection failed');
            }
        } catch (err) {
            saveStatusElement.textContent = `Error: ${err.message}`;
            saveStatusElement.classList.add('error');
        }
    });
    
    // --- Footer Year ---
    document.getElementById('currentYear').textContent = new Date().getFullYear();

    // --- Dark Mode Toggle ---
    const themeToggleCheckbox = document.getElementById('themeToggleCheckbox');
    const currentTheme = localStorage.getItem('theme');

    function setTheme(theme) {
        if (theme === 'dark') {
            document.body.classList.add('dark-mode');
            themeToggleCheckbox.checked = true;
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-mode');
            themeToggleCheckbox.checked = false;
            localStorage.setItem('theme', 'light');
        }
    }

    if (currentTheme) {
        setTheme(currentTheme);
    } else {
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (prefersDark) {
            setTheme('dark');
        } else {
            setTheme('light'); 
        }
    }

    themeToggleCheckbox.addEventListener('change', function() {
        localStorage.setItem('theme-has-been-set-by-user', 'true'); // User has made a choice
        if (this.checked) {
            setTheme('dark');
        } else {
            setTheme('light');
        }
    });

    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
        // Only apply OS theme if user hasn't explicitly set one via the toggle
        if (!localStorage.getItem('theme-has-been-set-by-user')) {
            const newColorScheme = e.matches ? "dark" : "light";
            setTheme(newColorScheme);
        }
    });

}); // End DOMContentLoaded