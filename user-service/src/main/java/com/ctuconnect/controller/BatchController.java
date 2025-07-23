package com.ctuconnect.controller;

import com.ctuconnect.dto.BatchDTO;
import com.ctuconnect.service.BatchService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/users/batches")
@RequiredArgsConstructor
public class BatchController {
    private final BatchService batchService;

    @GetMapping
    public ResponseEntity<List<BatchDTO>> getAllBatches() {
        List<BatchDTO> batches = batchService.getAllBatches();
        return ResponseEntity.ok(batches);
    }

    @GetMapping("/{year}")
    public ResponseEntity<BatchDTO> getBatchByYear(@PathVariable Integer year) {
        return batchService.getBatchByYear(year)
                .map(batch -> ResponseEntity.ok(batch))
                .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping
    public ResponseEntity<BatchDTO> createBatch(@RequestBody BatchDTO batchDTO) {
        BatchDTO createdBatch = batchService.createBatch(batchDTO);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdBatch);
    }

    @PutMapping("/{year}")
    public ResponseEntity<BatchDTO> updateBatch(@PathVariable Integer year, @RequestBody BatchDTO batchDTO) {
        return batchService.updateBatch(year, batchDTO)
                .map(batch -> ResponseEntity.ok(batch))
                .orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{year}")
    public ResponseEntity<Void> deleteBatch(@PathVariable Integer year) {
        if (batchService.deleteBatch(year)) {
            return ResponseEntity.noContent().build();
        }
        return ResponseEntity.notFound().build();
    }
}
